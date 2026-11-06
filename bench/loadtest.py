"""Gateway overhead and throughput under load.

Runs a mock OpenAI-compatible upstream with fixed latency, then drives identical closed-loop load
  1. straight at the upstream (baseline),
  2. through the gateway with the cache off (pure proxy overhead: auth, rate limit, routing, accounting),
  3. through the gateway with the semantic cache on, every request a miss (adds embed + store),
  4. through the gateway, every request an exact cache hit,
  5. through the gateway with paraphrased prompts: semantic hits, plus upstream calls where the guard says no.

    python -m bench.loadtest --duration 20 --procs 4 --concurrency 16
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import httpx

from bench.bodies import QUESTIONS, repeated
from bench.common import ROOT, gateway, load, mock_upstream

UP, GW = 9101, 8788


def gateway_config(cache: bool) -> str:
    return f"""
[cache]
enabled = {str(cache).lower()}
semantic = true
threshold = 0.94

[providers.mock]
type = "openai"
base_url = "http://127.0.0.1:{UP}/v1"
timeout = 30

[[models]]
name = "bench"
targets = [{{ provider = "mock", model = "mock-model" }}]
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=20)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--concurrency", type=int, default=16, help="in-flight requests per client process")
    ap.add_argument("--upstream-latency-ms", type=float, default=50)
    ap.add_argument("--out", default=str(ROOT / "eval" / "results" / "loadtest.json"))
    args = ap.parse_args()
    kw = dict(procs=args.procs, concurrency=args.concurrency, duration=args.duration)
    results: dict = {"config": {**vars(args), "platform": platform.platform(), "python": platform.python_version(),
                                "time": time.strftime("%Y-%m-%d %H:%M")}}

    def report(name: str, r: dict) -> None:
        results[name] = r
        lat = r["latency_ms"]
        print(f"{name:<22} {r['rps']:>8.1f} req/s   p50 {lat['p50']:>7} ms   p95 {lat['p95']:>7} ms   "
              f"p99 {lat['p99']:>7} ms   statuses {r['statuses']}   cache {r['cache']}", flush=True)
        if len(r["latency_by_cache"]) > 1:
            print(f"{'':<22} by cache result: {r['latency_by_cache']}", flush=True)

    with mock_upstream(UP, latency_ms=args.upstream_latency_ms):
        report("direct", load(f"http://127.0.0.1:{UP}/v1/chat/completions", {}, "unique_direct", **kw))

        with gateway(gateway_config(cache=False), GW) as key:
            h = {"Authorization": f"Bearer {key}"}
            report("gateway_no_cache", load(f"http://127.0.0.1:{GW}/v1/chat/completions", h, "unique", **kw))

        with gateway(gateway_config(cache=True), GW + 1) as key:
            h = {"Authorization": f"Bearer {key}"}
            url = f"http://127.0.0.1:{GW + 1}/v1/chat/completions"
            report("gateway_cache_miss", load(url, h, "unique", **kw))
            for q in QUESTIONS:  # warm the cache with the questions the hit scenarios ask
                httpx.post(url, json={**repeated(0, 0, 0), "messages": [{"role": "user", "content": q}]},
                           headers=h, timeout=30).raise_for_status()
            report("gateway_exact_hit", load(url, h, "repeated", **kw))
            report("gateway_semantic_hit", load(url, h, "paraphrased", **kw))

    d, g = results["direct"]["latency_ms"], results["gateway_no_cache"]["latency_ms"]
    results["overhead_ms"] = {k: round(g[k] - d[k], 2) for k in d}
    print("gateway overhead vs direct:", results["overhead_ms"])
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
