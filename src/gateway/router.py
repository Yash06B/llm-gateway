"""Fallback routing across an ordered list of targets, with a circuit breaker per target."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from .config import Config, Target
from .providers import Provider, UpstreamError


class Breaker:
    """Circuit breaker on the failure *rate* over a rolling window of recent calls.

    closed --(>= rate of the last `window` calls failed, with >= `min_calls` seen)--> open
    open --(cooldown elapsed)--> half-open: one probe --(ok)--> closed, window reset
                                                     \\--(fails)--> open again

    Its job is to stop paying for a dead or hung provider (every call to it costs a full timeout); fallback
    already absorbs a merely flaky one. So it trips on a high failure *rate* over a window, not a run of
    consecutive failures: under concurrency a provider failing 30% of requests produces runs of 3+ failures
    constantly, and a consecutive-failure breaker kept pulling it out of rotation (bench/chaos.py measured this).
    At 80% over 10+ calls, a 30%-flaky provider trips with probability ~0.2%; a dead one trips after 10 calls.
    """

    def __init__(self, window: int = 20, min_calls: int = 10, failure_rate: float = 0.8, cooldown: float = 30.0,
                 clock=time.monotonic):
        self.window = window
        self.min_calls = min_calls
        self.failure_rate = failure_rate
        self.cooldown = cooldown
        self.clock = clock
        self.state = "closed"
        self.outcomes: deque[bool] = deque(maxlen=window)  # True = failure
        self.opened_at = 0.0
        self._probe_in_flight = False

    @property
    def failures(self) -> int:
        return sum(self.outcomes)

    def allow(self) -> bool:
        if self.state == "closed":
            return True
        if self.state == "open" and self.clock() - self.opened_at >= self.cooldown:
            self.state = "half_open"
            self._probe_in_flight = False
        if self.state == "half_open" and not self._probe_in_flight:
            self._probe_in_flight = True
            return True
        return False

    def record_success(self) -> None:
        if self.state == "open":  # a request that was already in flight when the circuit opened
            return
        if self.state == "half_open":
            self.state, self._probe_in_flight = "closed", False
            self.outcomes.clear()
        self.outcomes.append(False)

    def record_failure(self) -> None:
        if self.state == "open":  # in flight when it opened: must not push the cooldown back
            return
        if self.state == "half_open":
            self._open()
            return
        self.outcomes.append(True)
        if len(self.outcomes) >= self.min_calls and self.failures / len(self.outcomes) >= self.failure_rate:
            self._open()

    def _open(self) -> None:
        self.state, self.opened_at, self._probe_in_flight = "open", self.clock(), False

    def retry_after(self) -> float:
        return max(0.0, self.cooldown - (self.clock() - self.opened_at)) if self.state == "open" else 0.0


@dataclass
class Attempt:
    target: str
    error: str | None = None
    status: int | None = None
    ms: float = 0.0


@dataclass
class RouteResult:
    target: Target
    attempts: list[Attempt] = field(default_factory=list)
    upstream_ms: float = 0.0  # time spent waiting on upstreams, including failed attempts
    ttft_ms: float | None = None  # streaming only


class NoRouteError(Exception):
    def __init__(self, message: str, attempts: list[Attempt], status: int = 502, retry_after: float = 0.0):
        super().__init__(message)
        self.attempts = attempts
        self.status = status
        self.retry_after = retry_after


class Router:
    def __init__(self, cfg: Config, providers: dict[str, Provider]):
        self.cfg = cfg
        self.providers = providers
        self.breakers: dict[str, Breaker] = {}

    def resolve(self, model: str) -> list[Target] | None:
        """A configured alias, or `provider/model` to address one upstream model directly."""
        if model in self.cfg.models:
            return self.cfg.models[model].targets
        provider, _, upstream = model.partition("/")
        if upstream and provider in self.providers:
            return [Target(provider, upstream)]
        return None

    def breaker(self, target: Target) -> Breaker:
        b = self.breakers.get(target.id)
        if b is None:
            r = self.cfg.routing
            b = self.breakers[target.id] = Breaker(r.window, r.min_calls, r.failure_rate, r.cooldown_seconds)
        return b

    def _candidates(self, targets: list[Target], attempts: list[Attempt]) -> list[Target]:
        out = []
        for t in targets:
            p = self.providers[t.provider]
            if not p.available:
                attempts.append(Attempt(t.id, "skipped: provider has no API key"))
            elif self.cfg.routing.breaker_enabled and not self.breaker(t).allow():
                attempts.append(Attempt(t.id, "skipped: circuit open"))
            else:
                out.append(t)
        return out

    def _no_route(self, targets: list[Target], attempts: list[Attempt], last: UpstreamError | None) -> NoRouteError:
        if last is None:  # nothing was even attempted: every circuit is open
            wait = min((self.breaker(t).retry_after() for t in targets), default=0.0)
            return NoRouteError("all upstream targets unavailable (circuits open)", attempts, 503, wait)
        return NoRouteError(f"all upstream targets failed; last error: {last}", attempts, 502)

    def _fail(self, target: Target, attempts: list[Attempt], e: UpstreamError, started: float) -> None:
        attempts.append(Attempt(target.id, str(e), e.status, (time.perf_counter() - started) * 1000))
        if e.retryable:  # a 400 is the client's fault, not a sign the upstream is unhealthy
            self.breaker(target).record_failure()

    async def complete(self, model: str, targets: list[Target], req: dict) -> tuple[dict, RouteResult]:
        attempts: list[Attempt] = []
        last: UpstreamError | None = None
        upstream_ms = 0.0
        for t in self._candidates(targets, attempts):
            started = time.perf_counter()
            try:
                body = await self.providers[t.provider].complete(t.model, req)
            except UpstreamError as e:
                upstream_ms += (time.perf_counter() - started) * 1000
                self._fail(t, attempts, e, started)
                if not e.retryable:
                    raise NoRouteError(str(e), attempts, e.status or 400) from e
                last = e
                continue
            ms = (time.perf_counter() - started) * 1000
            self.breaker(t).record_success()
            attempts.append(Attempt(t.id, ms=ms))
            return body, RouteResult(t, attempts, upstream_ms + ms)
        raise self._no_route(targets, attempts, last)

    async def stream(self, model: str, targets: list[Target], req: dict) -> tuple[AsyncIterator[dict], RouteResult]:
        """Fails over only until the first chunk arrives; after that the client already has partial output."""
        attempts: list[Attempt] = []
        last: UpstreamError | None = None
        upstream_ms = 0.0
        for t in self._candidates(targets, attempts):
            started = time.perf_counter()
            it = self.providers[t.provider].stream(t.model, req).__aiter__()
            try:
                first = await it.__anext__()
            except StopAsyncIteration:
                first = None
            except UpstreamError as e:
                upstream_ms += (time.perf_counter() - started) * 1000
                self._fail(t, attempts, e, started)
                if not e.retryable:
                    raise NoRouteError(str(e), attempts, e.status or 400) from e
                last = e
                continue
            ttft = (time.perf_counter() - started) * 1000
            attempts.append(Attempt(t.id, ms=ttft))
            result = RouteResult(t, attempts, upstream_ms, ttft_ms=upstream_ms + ttft)
            return self._relay(t, first, it, result, started), result
        raise self._no_route(targets, attempts, last)

    async def _relay(self, t: Target, first, it, result: RouteResult, started: float) -> AsyncIterator[dict]:
        try:
            if first is not None:
                yield first
            async for chunk in it:
                yield chunk
        except UpstreamError:
            self.breaker(t).record_failure()
            raise
        finally:
            result.upstream_ms += (time.perf_counter() - started) * 1000
        self.breaker(t).record_success()
