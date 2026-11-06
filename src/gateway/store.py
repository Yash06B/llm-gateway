"""SQLite persistence: API keys, the usage log and cache entries.

Usage rows are buffered in memory and flushed in batches by a background task, so a request never waits on disk.
"""

from __future__ import annotations

import asyncio
import hashlib
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass

SCHEMA = """
CREATE TABLE IF NOT EXISTS keys (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    prefix TEXT NOT NULL,
    hash TEXT NOT NULL UNIQUE,
    rpm INTEGER NOT NULL,
    tpm INTEGER NOT NULL,
    created REAL NOT NULL,
    disabled INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS usage (
    id INTEGER PRIMARY KEY,
    ts REAL NOT NULL,
    key_id INTEGER,
    model TEXT NOT NULL,
    target TEXT,
    status INTEGER NOT NULL,
    cache TEXT NOT NULL,
    stream INTEGER NOT NULL,
    prompt_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    cost_usd REAL NOT NULL,
    saved_usd REAL NOT NULL,
    latency_ms REAL NOT NULL,
    overhead_ms REAL,
    ttft_ms REAL,
    attempts INTEGER NOT NULL,
    error TEXT
);
CREATE INDEX IF NOT EXISTS usage_ts ON usage(ts);
CREATE INDEX IF NOT EXISTS usage_key ON usage(key_id, ts);
CREATE TABLE IF NOT EXISTS cache (
    key TEXT PRIMARY KEY,
    partition TEXT NOT NULL,
    query TEXT NOT NULL,
    embedding BLOB,
    body TEXT NOT NULL,
    cost_usd REAL NOT NULL,
    created REAL NOT NULL
);
"""

USAGE_COLUMNS = ("ts", "key_id", "model", "target", "status", "cache", "stream", "prompt_tokens",
                 "completion_tokens", "cost_usd", "saved_usd", "latency_ms", "overhead_ms", "ttft_ms",
                 "attempts", "error")


@dataclass
class ApiKey:
    id: int
    name: str
    prefix: str
    rpm: int
    tpm: int
    disabled: bool = False


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


class Store:
    def __init__(self, path: str):
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.executescript(SCHEMA)
        self._lock = threading.Lock()
        self._pending: list[tuple] = []
        self._keys: dict[str, ApiKey] = {}
        self._load_keys()

    # --- keys -------------------------------------------------------------------------------

    def _load_keys(self) -> None:
        rows = self._db.execute("SELECT id, name, prefix, hash, rpm, tpm, disabled FROM keys").fetchall()
        self._keys = {r[3]: ApiKey(r[0], r[1], r[2], r[4], r[5], bool(r[6])) for r in rows}

    def create_key(self, name: str, rpm: int, tpm: int) -> tuple[str, ApiKey]:
        """Returns the plaintext key once; only its hash is stored."""
        key = "gw-" + secrets.token_urlsafe(24)
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO keys (name, prefix, hash, rpm, tpm, created) VALUES (?, ?, ?, ?, ?, ?)",
                (name, key[:7], hash_key(key), rpm, tpm, time.time()),
            )
            api_key = ApiKey(cur.lastrowid, name, key[:7], rpm, tpm)
            self._keys[hash_key(key)] = api_key
        return key, api_key

    def authenticate(self, key: str) -> ApiKey | None:
        k = self._keys.get(hash_key(key))
        return k if k and not k.disabled else None

    def list_keys(self) -> list[ApiKey]:
        return sorted(self._keys.values(), key=lambda k: k.id)

    # --- usage ------------------------------------------------------------------------------

    def log_usage(self, **row) -> None:
        self._pending.append(tuple(row.get(c) for c in USAGE_COLUMNS))

    def flush(self) -> int:
        rows, self._pending = self._pending, []
        if rows:
            with self._lock:
                self._db.executemany(
                    f"INSERT INTO usage ({', '.join(USAGE_COLUMNS)}) VALUES ({', '.join('?' * len(USAGE_COLUMNS))})",
                    rows,
                )
        return len(rows)

    async def flush_forever(self, interval: float = 0.5) -> None:
        while True:
            await asyncio.sleep(interval)
            if self._pending:
                await asyncio.to_thread(self.flush)

    def usage_summary(self, since: float, group_by: str) -> list[dict]:
        col = {"key": "key_id", "model": "model", "target": "target", "cache": "cache"}[group_by]
        self.flush()
        with self._lock:
            rows = self._db.execute(
                f"""SELECT {col}, COUNT(*), SUM(status < 400), SUM(prompt_tokens), SUM(completion_tokens),
                           SUM(cost_usd), SUM(saved_usd), AVG(latency_ms)
                    FROM usage WHERE ts >= ? GROUP BY {col} ORDER BY COUNT(*) DESC""",
                (since,),
            ).fetchall()
        names = {k.id: k.name for k in self._keys.values()}
        return [
            {
                group_by: names.get(r[0], r[0]) if group_by == "key" else r[0],
                "requests": r[1], "ok": r[2], "prompt_tokens": r[3], "completion_tokens": r[4],
                "cost_usd": round(r[5] or 0, 6), "saved_usd": round(r[6] or 0, 6),
                "avg_latency_ms": round(r[7] or 0, 1),
            }
            for r in rows
        ]

    # --- cache ------------------------------------------------------------------------------

    def cache_put(self, key: str, partition: str, query: str, embedding: bytes | None, body: str,
                  cost_usd: float) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO cache VALUES (?, ?, ?, ?, ?, ?, ?)",
                (key, partition, query, embedding, body, cost_usd, time.time()),
            )

    def cache_delete(self, keys: list[str]) -> None:
        with self._lock:
            self._db.executemany("DELETE FROM cache WHERE key = ?", [(k,) for k in keys])

    def cache_load(self, since: float) -> list[tuple]:
        with self._lock:
            self._db.execute("DELETE FROM cache WHERE created < ?", (since,))
            return self._db.execute(
                "SELECT key, partition, query, embedding, body, cost_usd, created FROM cache ORDER BY created"
            ).fetchall()

    def close(self) -> None:
        self.flush()
        self._db.close()
