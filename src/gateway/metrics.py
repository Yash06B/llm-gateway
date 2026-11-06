"""In-process metrics: counters plus a rolling window of recent requests for percentiles and rates.

Exposed as JSON for the dashboard (`/admin/stats`) and in Prometheus text format (`/metrics`).
"""

from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass

import numpy as np


@dataclass
class Sample:
    ts: float
    model: str
    target: str | None
    status: int
    cache: str
    latency_ms: float
    overhead_ms: float | None
    ttft_ms: float | None
    cost_usd: float
    saved_usd: float


def _pct(values: list[float]) -> dict:
    if not values:
        return {"p50": None, "p95": None, "p99": None}
    a = np.asarray(values)
    return {f"p{q}": round(float(np.percentile(a, q)), 2) for q in (50, 95, 99)}


class Metrics:
    def __init__(self, window: int = 20_000):
        self.started = time.time()
        self.recent: deque[Sample] = deque(maxlen=window)
        self.requests = Counter()  # (model, target, status class, cache) -> n
        self.tokens = Counter()  # (target, "prompt"|"completion") -> n
        self.cost_usd = 0.0
        self.saved_usd = 0.0
        self.fallbacks = 0
        self.rate_limited = 0

    def record(self, s: Sample, prompt_tokens: int = 0, completion_tokens: int = 0, attempts: int = 1) -> None:
        self.recent.append(s)
        self.requests[(s.model, s.target or "-", f"{s.status // 100}xx", s.cache)] += 1
        if s.target:
            self.tokens[(s.target, "prompt")] += prompt_tokens
            self.tokens[(s.target, "completion")] += completion_tokens
        self.cost_usd += s.cost_usd
        self.saved_usd += s.saved_usd
        if attempts > 1:
            self.fallbacks += 1

    def snapshot(self, window_s: float = 300) -> dict:
        now = time.time()
        recent = [s for s in self.recent if now - s.ts <= window_s]
        last_min = [s for s in recent if now - s.ts <= 60]
        ok = [s for s in recent if s.status < 400]
        misses = [s for s in ok if s.cache == "miss"]
        hits = [s for s in ok if s.cache in ("exact", "semantic")]
        by_target: dict[str, list[Sample]] = {}
        for s in misses:
            by_target.setdefault(s.target or "-", []).append(s)
        return {
            "uptime_s": round(now - self.started),
            "window_s": window_s,
            "requests": len(recent),
            "rps_1m": round(len(last_min) / 60, 2),
            "error_rate": round(1 - len(ok) / len(recent), 4) if recent else 0.0,
            "cache_hit_rate": round(len(hits) / len(ok), 4) if ok else 0.0,
            "cache_hits": {k: sum(1 for s in hits if s.cache == k) for k in ("exact", "semantic")},
            "latency_ms": {
                "miss": _pct([s.latency_ms for s in misses]),
                "hit": _pct([s.latency_ms for s in hits]),
                "overhead": _pct([s.overhead_ms for s in misses if s.overhead_ms is not None]),
                "ttft": _pct([s.ttft_ms for s in misses if s.ttft_ms is not None]),
            },
            "targets": {
                t: {"requests": len(v), **_pct([s.latency_ms for s in v])} for t, v in sorted(by_target.items())
            },
            "totals": {
                "requests": sum(self.requests.values()),
                "cost_usd": round(self.cost_usd, 6),
                "saved_usd": round(self.saved_usd, 6),
                "fallbacks": self.fallbacks,
                "rate_limited": self.rate_limited,
            },
            "series": self._series(recent, now, window_s),
        }

    @staticmethod
    def _series(recent: list[Sample], now: float, window_s: float, buckets: int = 30) -> list[dict]:
        width = window_s / buckets
        out = [{"t": round(now - window_s + (i + 1) * width), "requests": 0, "hits": 0, "errors": 0}
               for i in range(buckets)]
        for s in recent:
            i = min(buckets - 1, int((s.ts - (now - window_s)) / width))
            out[i]["requests"] += 1
            out[i]["hits"] += s.cache in ("exact", "semantic")
            out[i]["errors"] += s.status >= 400
        return out

    def prometheus(self, breakers: dict) -> str:
        lines = ["# TYPE gateway_requests_total counter"]
        for (model, target, status, cache), n in sorted(self.requests.items()):
            lines.append(f'gateway_requests_total{{model="{model}",target="{target}",status="{status}",'
                         f'cache="{cache}"}} {n}')
        lines.append("# TYPE gateway_tokens_total counter")
        for (target, kind), n in sorted(self.tokens.items()):
            lines.append(f'gateway_tokens_total{{target="{target}",kind="{kind}"}} {n}')
        lines += [
            "# TYPE gateway_cost_usd_total counter", f"gateway_cost_usd_total {self.cost_usd:.6f}",
            "# TYPE gateway_saved_usd_total counter", f"gateway_saved_usd_total {self.saved_usd:.6f}",
            "# TYPE gateway_fallbacks_total counter", f"gateway_fallbacks_total {self.fallbacks}",
            "# TYPE gateway_rate_limited_total counter", f"gateway_rate_limited_total {self.rate_limited}",
            "# TYPE gateway_latency_ms summary",
        ]
        lat = [s.latency_ms for s in self.recent if s.status < 400]
        for q, v in _pct(lat).items():
            if v is not None:
                lines.append(f'gateway_latency_ms{{quantile="0.{q[1:]}"}} {v}')
        lines.append("# TYPE gateway_circuit_open gauge")
        for target, b in sorted(breakers.items()):
            lines.append(f'gateway_circuit_open{{target="{target}"}} {int(b.state != "closed")}')
        return "\n".join(lines) + "\n"
