"""What fallback routing and circuit breakers buy when a provider misbehaves.

Two mock upstreams: `primary` (the one being broken) and `backup` (healthy, 50 ms).

  flaky  primary fails 30% of requests with a 503.
         - "solo" alias (primary only) vs "chaos" alias (primary, then backup)
  down   primary accepts connections but never answers (a hung upstream); the gateway's timeout is 1 s.
         - fallback without a circuit breaker: every request waits out the timeout before failing over
         - fallback with a circuit breaker: once 80% of recent calls have timed out, the circuit opens
           and requests skip primary (one probe per 5 s cooldown)

    python -m bench.chaos --duration 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bench.common import ROOT, gateway, load, mock_upstream

PRIMARY, BACKUP, GW = 9201, 9202, 8790


def config(breaker: bool, primary_timeout: float) -> str:
    return f"""
[cache]
enabled = false

[routing]
cooldown_seconds = 5
breaker_enabled = {str(breaker).lower()}

[providers.primary]
type = "openai"
base_url = "http://127.0.0.1:{PRIMARY}/v1"
timeout = {primary_timeout}

[providers.backup]
type = "openai"
base_url = "http://127.0.0.1:{BACKUP}/v1"
timeout = 30

[[models]]
name = "chaos"
targets = [{{ provider = "primary", model = "p" }}, {{ provider = "backup", model = "b" }}]

[[models]]
name = "solo"
targets = [{{ provider = "primary", model = "p" }}]
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=20)
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "eval" / "results" / "chaos.json"))
    args = ap.parse_args()
    kw = dict(procs=args.procs, concurrency=args.concurrency, duration=args.duration)
    results: dict = {"config": vars(args)}

    def run(name: str, port: int, key: str, body: str) -> None:
        r = load(f"http://127.0.0.1:{port}/v1/chat/completions", {"Authorization": f"Bearer {key}"}, body, **kw)
        r["success_rate"] = round(r["ok"] / r["requests"], 4) if r["requests"] else 0.0
        results[name] = r
        lat = r["latency_ms"]
        print(f"{name:<28} success {r['success_rate']:>7.2%}   {r['rps']:>7.1f} ok/s   p50 {lat['p50']:>8} ms   "
              f"p95 {lat['p95']:>8} ms   p99 {lat['p99']:>8} ms   targets {r['targets']}", flush=True)

    with mock_upstream(BACKUP, latency_ms=50):
        with mock_upstream(PRIMARY, latency_ms=50, fail_rate=0.3, seed=1):
            with gateway(config(breaker=True, primary_timeout=30), GW) as key:
                run("flaky_no_fallback", GW, key, "chaos_solo")
                run("flaky_with_fallback", GW, key, "chaos")

        with mock_upstream(PRIMARY, latency_ms=600_000):  # hung: accepts the request, never answers
            with gateway(config(breaker=False, primary_timeout=1), GW + 1) as key:
                run("down_fallback_no_breaker", GW + 1, key, "chaos")
            with gateway(config(breaker=True, primary_timeout=1), GW + 2) as key:
                run("down_fallback_with_breaker", GW + 2, key, "chaos")

    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
