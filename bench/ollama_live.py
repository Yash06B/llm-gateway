"""End to end with a real local model: what a cache hit is worth against Ollama.

Asks 20 questions through the gateway (upstream calls), then asks them again (exact hits), then asks paraphrases
of them (semantic hits where the cache recognises them), then asks near-miss variants that must NOT be served from
cache. Requires Ollama running locally with the model pulled.

    python -m bench.ollama_live --model llama3.2:3b
"""

from __future__ import annotations

import argparse
import json
import time

import httpx

from bench.common import ROOT, gateway, percentiles
from eval.cache_pairs import HELDOUT_DIFFERENT, HELDOUT_SAME

GW = 8795


def config(model: str) -> str:
    return f"""
[providers.ollama]
type = "openai"
base_url = "http://127.0.0.1:11434/v1"
timeout = 300

[[models]]
name = "local"
targets = [{{ provider = "ollama", model = "{model}" }}]
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.2:3b")
    args = ap.parse_args()

    originals = [a for _, a, _ in HELDOUT_SAME]
    paraphrases = [b for _, _, b in HELDOUT_SAME]
    # near-misses whose first prompt is one of the originals, so a wrong hit is possible
    near = [b for _, a, b in HELDOUT_DIFFERENT if a in originals]

    with gateway(config(args.model), GW) as key:
        client = httpx.Client(base_url=f"http://127.0.0.1:{GW}", timeout=300,
                              headers={"Authorization": f"Bearer {key}"})

        def ask(q: str) -> tuple[float, str, str]:
            t = time.perf_counter()
            r = client.post("/v1/chat/completions", json={
                "model": "local", "temperature": 0, "max_tokens": 200,
                "messages": [{"role": "user", "content": q}]})
            r.raise_for_status()
            return (time.perf_counter() - t) * 1000, r.headers.get("x-gateway-cache", "?"), \
                r.json()["choices"][0]["message"]["content"]

        ask("Say OK.")  # load the model so the first question doesn't pay for it
        rounds = {}
        for name, qs in (("first_ask", originals), ("repeat", originals), ("paraphrase", paraphrases),
                         ("near_miss", near)):
            rows = [ask(q) for q in qs]
            kinds = {k: sum(1 for _, c, _ in rows if c == k) for k in sorted({c for _, c, _ in rows})}
            rounds[name] = {"n": len(rows), "cache": kinds, "latency_ms": percentiles([ms for ms, _, _ in rows]),
                            "hit_latency_ms": percentiles([ms for ms, c, _ in rows if c in ("exact", "semantic")]),
                            "miss_latency_ms": percentiles([ms for ms, c, _ in rows if c == "miss"])}
            print(f"{name:<11} {kinds}  latency {rounds[name]['latency_ms']}", flush=True)
        stats = client.get("/admin/stats", headers={"Authorization": "Bearer bench-admin"}).json()

    result = {"model": args.model, "rounds": rounds, "cache": stats["cache"]}
    path = ROOT / "eval" / "results" / "ollama_live.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("wrote", path)


if __name__ == "__main__":
    main()
