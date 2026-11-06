"""Two micro-benchmarks behind design decisions in the gateway.

1. HTTP client: one shared httpx.AsyncClient vs one aiohttp.ClientSession, driving the same mock upstream (50 ms)
   from a single process at increasing concurrency. This is the gateway's situation: one process, one client,
   every request in flight at once going through it.
2. Embedding batch size: texts per second for the cache's embedder at batch sizes 1..64 (why EmbedBatcher exists).

    python -m bench.micro
"""

from __future__ import annotations

import asyncio
import json
import time

import aiohttp
import httpx
import numpy as np

from bench.bodies import VOCAB
from bench.common import ROOT, mock_upstream

UP = 9301
URL = f"http://127.0.0.1:{UP}/v1/chat/completions"


async def drive(post, concurrency: int, duration: float) -> dict:
    lat: list[float] = []
    deadline = time.perf_counter() + duration

    async def one(slot: int) -> None:
        i = 0
        while time.perf_counter() < deadline:
            i += 1
            t = time.perf_counter()
            await post({"model": "m", "messages": [{"role": "user", "content": f"{slot} {i}"}]})
            lat.append((time.perf_counter() - t) * 1000)

    await asyncio.gather(*(one(s) for s in range(concurrency)))
    return {"rps": round(len(lat) / duration, 1), "p50": round(float(np.percentile(lat, 50)), 1),
            "p99": round(float(np.percentile(lat, 99)), 1)}


async def with_httpx(concurrency: int, duration: float) -> dict:
    limits = httpx.Limits(max_connections=1024, max_keepalive_connections=1024)
    async with httpx.AsyncClient(timeout=60, limits=limits) as c:
        async def post(body):
            (await c.post(URL, json=body)).json()

        return await drive(post, concurrency, duration)


async def with_aiohttp(concurrency: int, duration: float) -> dict:
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=1024)) as s:
        async def post(body):
            async with s.post(URL, json=body) as r:
                await r.json()

        return await drive(post, concurrency, duration)


def clients(duration: float = 6) -> dict:
    out = {}
    with mock_upstream(UP, latency_ms=50):
        for c in (16, 64, 128):
            for name, fn in (("httpx", with_httpx), ("aiohttp", with_aiohttp)):
                out[f"{name}@{c}"] = r = asyncio.run(fn(c, duration))
                print(f"{name:<8} concurrency {c:>4}: {r['rps']:>7} req/s   p50 {r['p50']:>7} ms   p99 {r['p99']:>8} ms",
                      flush=True)
    return out


def embedding(n: int = 1024) -> dict:
    from gateway.cache import Embedder

    rng = np.random.default_rng(0)
    texts = [" ".join(rng.choice(VOCAB, 10)) for _ in range(n)]
    emb = Embedder("BAAI/bge-small-en-v1.5")
    emb.embed(texts[:64])  # warm up
    out = {}
    for bs in (1, 4, 16, 64):
        t = time.perf_counter()
        for i in range(0, n, bs):
            emb.embed(texts[i:i + bs])
        out[bs] = round(n / (time.perf_counter() - t), 1)
        print(f"embed batch {bs:>3}: {out[bs]:>8} texts/s", flush=True)
    return out


def main() -> None:
    result = {"http_clients": clients(), "embed_texts_per_s_by_batch": embedding()}
    path = ROOT / "eval" / "results" / "micro.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("wrote", path)


if __name__ == "__main__":
    main()
