"""Shared helpers for the benchmarks: start servers as subprocesses and drive closed-loop load."""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import os
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import aiohttp
import httpx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def wait_ready(url: str, timeout: float = 60) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if httpx.get(url, timeout=1).status_code < 500:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.2)
    raise RuntimeError(f"{url} not ready after {timeout}s")


@contextmanager
def server(args: list[str], ready_url: str, env: dict | None = None, log: str | None = None):
    out = open(ROOT / "bench" / log, "w") if log else subprocess.DEVNULL
    proc = subprocess.Popen([sys.executable, "-m", "gateway.cli", *args], cwd=ROOT, stdout=out, stderr=subprocess.STDOUT,
                            env={**os.environ, "HF_HUB_DISABLE_SYMLINKS_WARNING": "1", **(env or {})})
    try:
        wait_ready(ready_url)
        yield proc
    finally:
        proc.terminate()
        proc.wait(10)
        if log:
            out.close()


@contextmanager
def mock_upstream(port: int, latency_ms: float = 0, fail_rate: float = 0, seed: int = 0):
    args = ["mock-upstream", "--port", str(port), "--latency-ms", str(latency_ms), "--fail-rate", str(fail_rate),
            "--seed", str(seed)]
    with server(args, f"http://127.0.0.1:{port}/stats") as p:
        yield p


@contextmanager
def gateway(config_toml: str, port: int, admin_key: str = "bench-admin"):
    """Writes the config to a temp dir, creates an unlimited key, and runs `gateway serve`. Yields the API key."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:  # Windows may still hold gw.db
        cfg = Path(d) / "gateway.toml"
        cfg.write_text(f'[server]\ndb_path = "{(Path(d) / "gw.db").as_posix()}"\n\n{config_toml}', encoding="utf-8")
        out = subprocess.run([sys.executable, "-m", "gateway.cli", "keys", "--config", str(cfg), "create", "bench",
                              "--rpm", "100000000", "--tpm", "1000000000"], cwd=ROOT, capture_output=True, text=True,
                             check=True)
        key = out.stdout.splitlines()[1].strip()
        with server(["serve", "--config", str(cfg), "--port", str(port)], f"http://127.0.0.1:{port}/healthz",
                    env={"GATEWAY_ADMIN_KEY": admin_key}, log=f"gateway-{port}.log"):
            yield key


def percentiles(ms: list[float]) -> dict:
    if not ms:
        return {"p50": None, "p95": None, "p99": None}
    a = np.asarray(ms)
    return {f"p{q}": round(float(np.percentile(a, q)), 2) for q in (50, 95, 99)}


async def _worker_loop(url, headers, make_body, concurrency, duration, worker, results):
    # aiohttp, not httpx: a single httpx AsyncClient caps out far below what this needs (see the README)
    timeout = aiohttp.ClientTimeout(total=60)
    async with aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(limit=0)) as client:
        deadline = time.perf_counter() + duration

        async def one(slot):
            i = 0
            while time.perf_counter() < deadline:
                body = make_body(worker, slot, i)
                i += 1
                t = time.perf_counter()
                try:
                    async with client.post(url, json=body, headers=headers) as r:
                        await r.read()
                        status = r.status
                        meta = {k: r.headers.get(k) for k in ("x-gateway-cache", "x-gateway-target")}
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    status, meta = 0, {"error": type(e).__name__}
                results.append(((time.perf_counter() - t) * 1000, status, meta))

        await asyncio.gather(*(one(s) for s in range(concurrency)))


def _proc_main(args):
    url, headers, body_fn_name, concurrency, duration, worker = args
    import bench.bodies as bodies

    results: list = []
    asyncio.run(_worker_loop(url, headers, getattr(bodies, body_fn_name), concurrency, duration, worker, results))
    return results


def load(url: str, headers: dict, body_fn: str, *, procs: int, concurrency: int, duration: float) -> dict:
    """Closed-loop load: `procs` client processes x `concurrency` in-flight requests each, for `duration` seconds."""
    started = time.perf_counter()
    with mp.get_context("spawn").Pool(procs) as pool:
        parts = pool.map(_proc_main, [(url, headers, body_fn, concurrency, duration, w) for w in range(procs)])
    elapsed = time.perf_counter() - started
    rows = [r for part in parts for r in part]
    ok = [r for r in rows if r[1] == 200]
    return {
        "requests": len(rows),
        "ok": len(ok),
        "rps": round(len(ok) / duration, 1),
        "latency_ms": percentiles([r[0] for r in ok]),
        "statuses": {str(s): sum(1 for r in rows if r[1] == s) for s in sorted({r[1] for r in rows})},
        "cache": {str(c): sum(1 for r in ok if str(r[2].get("x-gateway-cache")) == c)
                  for c in sorted({str(r[2].get("x-gateway-cache")) for r in ok})},
        "latency_by_cache": {str(c): percentiles([r[0] for r in ok if str(r[2].get("x-gateway-cache")) == c])
                             for c in sorted({str(r[2].get("x-gateway-cache")) for r in ok})},
        "targets": {str(t): sum(1 for r in ok if str(r[2].get("x-gateway-target")) == t)
                    for t in sorted({str(r[2].get("x-gateway-target")) for r in ok})},
        "wall_s": round(elapsed, 1),
    }
