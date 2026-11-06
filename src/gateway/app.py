"""The FastAPI app: an OpenAI-compatible `/v1/chat/completions` plus admin, metrics and a dashboard."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse

from .cache import CacheKeys, Hit, ResponseCache, cache_keys
from .config import Config
from .metrics import Metrics, Sample
from .providers import (Provider, UpstreamError, estimate_tokens, make_chunk, make_completion, make_provider,
                        make_usage_chunk, message_text, new_id)
from .ratelimit import RateLimiter
from .router import NoRouteError, RouteResult, Router
from .store import ApiKey, Store

log = logging.getLogger("gateway")


def error(status: int, message: str, kind: str = "invalid_request_error", headers: dict | None = None,
          **extra) -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": kind, **extra}}, status, headers=headers)


class Gateway:
    def __init__(self, cfg: Config, providers: dict[str, Provider] | None = None, embedder=None,
                 admin_key: str | None = None):
        self.cfg = cfg
        self.store = Store(cfg.db_path)
        self.providers = providers or {name: make_provider(p) for name, p in cfg.providers.items()}
        self.router = Router(cfg, self.providers)
        self.cache = ResponseCache(cfg.cache, self.store, embedder) if cfg.cache.enabled else None
        self.limiter = RateLimiter()
        self.metrics = Metrics()
        self.admin_key = admin_key or os.environ.get("GATEWAY_ADMIN_KEY") or ""

    def finish(self, *, key: ApiKey, model: str, started: float, status: int, cache: str, stream: bool,
               route: RouteResult | None = None, prompt_tokens: int = 0, completion_tokens: int = 0,
               saved_usd: float = 0.0, error_msg: str | None = None) -> float:
        latency = (time.perf_counter() - started) * 1000
        target = route.target.id if route else None
        cost = self.cfg.cost(target, prompt_tokens, completion_tokens) if target and cache == "miss" else 0.0
        overhead = latency - route.upstream_ms if route and status < 400 and not stream else None
        ttft = route.ttft_ms if route else None
        attempts = len([a for a in route.attempts if not (a.error or "").startswith("skipped")]) if route else 0
        self.metrics.record(Sample(time.time(), model, target, status, cache, latency, overhead, ttft, cost,
                                   saved_usd), prompt_tokens, completion_tokens, attempts)
        self.store.log_usage(ts=time.time(), key_id=key.id, model=model, target=target, status=status, cache=cache,
                             stream=int(stream), prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                             cost_usd=cost, saved_usd=saved_usd, latency_ms=latency, overhead_ms=overhead,
                             ttft_ms=ttft, attempts=attempts, error=error_msg)
        if prompt_tokens or completion_tokens:
            self.limiter.charge_tokens(key.id, key.rpm, key.tpm, prompt_tokens + completion_tokens)
        return cost


def _route_headers(route: RouteResult) -> dict:
    return {"x-gateway-target": route.target.id,
            "x-gateway-attempts": str(len([a for a in route.attempts if not (a.error or "").startswith("skipped")]))}


def _sse(obj) -> str:
    return f"data: {json.dumps(obj, separators=(',', ':'))}\n\n"


def _replay_stream(body: dict, model: str, include_usage: bool):
    """Serve a cached completion as a stream: one content chunk, then the finish chunk."""
    cid, choice = new_id(), body["choices"][0]
    yield _sse(make_chunk(cid, model, {"role": "assistant", "content": choice["message"].get("content") or ""}))
    yield _sse(make_chunk(cid, model, {}, choice.get("finish_reason") or "stop"))
    if include_usage:
        yield _sse(make_usage_chunk(cid, model, 0, 0))
    yield "data: [DONE]\n\n"


def create_app(cfg: Config, gateway: Gateway | None = None) -> FastAPI:
    gw = gateway or Gateway(cfg)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not gw.admin_key:
            gw.admin_key = "gwadmin-" + secrets.token_urlsafe(16)
            log.warning("GATEWAY_ADMIN_KEY not set; generated admin key for this run: %s", gw.admin_key)
        flusher = asyncio.create_task(gw.store.flush_forever())
        yield
        flusher.cancel()
        for p in gw.providers.values():
            await p.aclose()
        gw.store.close()

    app = FastAPI(title="LLM Gateway", version="0.1.0", lifespan=lifespan)
    app.state.gateway = gw

    def bearer(request: Request) -> str:
        auth = request.headers.get("authorization", "")
        return auth[7:].strip() if auth.lower().startswith("bearer ") else request.headers.get("x-api-key", "")

    def is_admin(request: Request) -> bool:
        return bool(gw.admin_key) and secrets.compare_digest(bearer(request), gw.admin_key)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        started = time.perf_counter()
        key = gw.store.authenticate(bearer(request))
        if key is None:
            return error(401, "invalid or missing API key", "authentication_error")
        try:
            req = await request.json()
        except json.JSONDecodeError:
            return error(400, "request body is not valid JSON")
        model, messages = req.get("model"), req.get("messages")
        if not isinstance(model, str) or not isinstance(messages, list) or not messages:
            return error(400, "`model` (string) and `messages` (non-empty list) are required")

        decision = gw.limiter.check(key.id, key.rpm, key.tpm)
        if not decision.allowed:
            gw.metrics.rate_limited += 1
            return error(429, f"rate limit exceeded ({decision.limit} per minute)", "rate_limit_error",
                         headers={**decision.headers, "retry-after": str(math.ceil(decision.retry_after))})
        headers = dict(decision.headers)

        targets = gw.router.resolve(model)
        if targets is None:
            return error(404, f"unknown model {model!r}; see GET /v1/models", "not_found_error")

        stream = bool(req.get("stream"))
        include_usage = bool((req.get("stream_options") or {}).get("include_usage"))
        mode = request.headers.get("x-gateway-cache")
        keys: CacheKeys | None = None
        if gw.cache and gw.cache.cacheable(req, mode):
            keys = cache_keys(model, req)
        if keys:
            hit = await gw.cache.get(keys, semantic=mode != "exact")
            if hit:
                return serve_hit(hit, key, model, req, stream, include_usage, started, headers)
            headers["x-gateway-cache"] = "miss"

        try:
            if stream:
                it, route = await gw.router.stream(model, targets, req)
            else:
                body, route = await gw.router.complete(model, targets, req)
        except NoRouteError as e:
            gw.finish(key=key, model=model, started=started, status=e.status, cache="miss", stream=stream,
                      error_msg=str(e))
            h = {"retry-after": str(math.ceil(e.retry_after))} if e.retry_after else None
            return error(e.status, str(e), "upstream_error", headers=h,
                         attempts=[a.__dict__ for a in e.attempts])
        headers.update(_route_headers(route))

        if not stream:
            usage = body.get("usage") or {}
            pt = usage.get("prompt_tokens") or sum(estimate_tokens(message_text(m.get("content"))) for m in messages)
            ct = usage.get("completion_tokens") or estimate_tokens(
                message_text(body["choices"][0]["message"].get("content")) if body.get("choices") else "")
            cost = gw.finish(key=key, model=model, started=started, status=200, cache="miss", stream=False,
                             route=route, prompt_tokens=pt, completion_tokens=ct)
            if keys and body.get("choices") and body["choices"][0].get("finish_reason") in ("stop", "length"):
                await gw.cache.put(keys, body, cost)
            return JSONResponse(body, headers=headers)

        return StreamingResponse(relay(it, route, key, model, req, keys, include_usage, started),
                                 media_type="text/event-stream", headers=headers)

    def serve_hit(hit: Hit, key: ApiKey, model: str, req: dict, stream: bool, include_usage: bool, started: float,
                  headers: dict):
        headers = {**headers, "x-gateway-cache": hit.kind, "x-gateway-similarity": f"{hit.similarity:.4f}"}
        gw.finish(key=key, model=model, started=started, status=200, cache=hit.kind, stream=stream,
                  saved_usd=hit.cost_usd)
        if stream:
            return StreamingResponse(_replay_stream(hit.body, model, include_usage), media_type="text/event-stream",
                                     headers=headers)
        body = {**hit.body, "id": new_id(), "created": int(time.time()),
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}
        return JSONResponse(body, headers=headers)

    async def relay(it, route: RouteResult, key: ApiKey, model: str, req: dict, keys: CacheKeys | None,
                    include_usage: bool, started: float):
        parts: list[str] = []
        pt = ct = 0
        finish_reason = None
        status, err = 200, None
        try:
            async for chunk in it:
                if chunk.get("usage"):
                    pt = chunk["usage"].get("prompt_tokens") or pt
                    ct = chunk["usage"].get("completion_tokens") or ct
                    if not chunk.get("choices") and not include_usage:
                        continue  # the gateway asked for usage; the client didn't
                for c in chunk.get("choices") or []:
                    parts.append((c.get("delta") or {}).get("content") or "")
                    finish_reason = c.get("finish_reason") or finish_reason
                yield _sse(chunk)
            yield "data: [DONE]\n\n"
        except UpstreamError as e:
            status, err = 502, str(e)
            yield _sse({"error": {"message": f"upstream failed mid-stream: {e}", "type": "upstream_error"}})
        except (asyncio.CancelledError, GeneratorExit):
            status, err = 499, "client disconnected"
            raise
        finally:
            text = "".join(parts)
            pt = pt or sum(estimate_tokens(message_text(m.get("content"))) for m in req["messages"])
            ct = ct or (estimate_tokens(text) if text else 0)
            cost = gw.finish(key=key, model=model, started=started, status=status, cache="miss", stream=True,
                             route=route, prompt_tokens=pt, completion_tokens=ct, error_msg=err)
        if keys and status == 200 and finish_reason in ("stop", "length"):
            await gw.cache.put(keys, make_completion(route.target.model, text, finish_reason, pt, ct), cost)

    @app.get("/v1/models")
    async def list_models(request: Request):
        if gw.store.authenticate(bearer(request)) is None and not is_admin(request):
            return error(401, "invalid or missing API key", "authentication_error")
        data = [{"id": name, "object": "model", "owned_by": "gateway",
                 "targets": [t.id for t in route.targets]} for name, route in cfg.models.items()]
        return {"object": "list", "data": data}

    @app.get("/healthz")
    async def healthz():
        return {"ok": True}

    @app.get("/metrics")
    async def metrics(request: Request):
        if not is_admin(request):
            return error(401, "admin key required", "authentication_error")
        return PlainTextResponse(gw.metrics.prometheus(gw.router.breakers))

    @app.get("/admin/stats")
    async def stats(request: Request, window: float = 300):
        if not is_admin(request):
            return error(401, "admin key required", "authentication_error")
        snap = gw.metrics.snapshot(window)
        snap["breakers"] = {t: {"state": b.state, "failures": b.failures, "retry_after_s": round(b.retry_after(), 1)}
                            for t, b in sorted(gw.router.breakers.items())}
        snap["cache"] = gw.cache.stats() if gw.cache else {"enabled": False}
        snap["providers"] = {n: {"available": p.available} for n, p in gw.providers.items()}
        return snap

    @app.get("/admin/usage")
    async def usage(request: Request, hours: float = 24, group_by: str = "key"):
        if not is_admin(request):
            return error(401, "admin key required", "authentication_error")
        if group_by not in ("key", "model", "target", "cache"):
            return error(400, "group_by must be one of key, model, target, cache")
        rows = await asyncio.to_thread(gw.store.usage_summary, time.time() - hours * 3600, group_by)
        return {"hours": hours, "group_by": group_by, "rows": rows}

    @app.get("/admin/keys")
    async def list_keys(request: Request):
        if not is_admin(request):
            return error(401, "admin key required", "authentication_error")
        return {"keys": [k.__dict__ for k in gw.store.list_keys()]}

    @app.post("/admin/keys")
    async def create_key(request: Request):
        if not is_admin(request):
            return error(401, "admin key required", "authentication_error")
        body = await request.json()
        key, rec = gw.store.create_key(body.get("name", "unnamed"), int(body.get("rpm", cfg.limits.default_rpm)),
                                       int(body.get("tpm", cfg.limits.default_tpm)))
        return {"key": key, **rec.__dict__}

    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard():
        return (Path(__file__).parent / "web" / "dashboard.html").read_text(encoding="utf-8")

    return app
