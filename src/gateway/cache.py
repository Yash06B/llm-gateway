"""Exact and semantic response cache.

A request is split into
  - the *context*: model, system prompt, every message before the last user turn, and the sampling params, and
  - the *query*: the text of the last user message.

Exact hits need the whole request to match. Semantic hits need the context to match exactly (it becomes the
partition key) and the query to be close in embedding space. Matching only the last turn inside an identical
context is what keeps "same question, different conversation" from returning the wrong answer.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from . import guard
from .config import CacheConfig
from .providers import message_text
from .store import Store

# Params that change the answer. Anything else (user, stream, stream_options, ...) is ignored for keying.
KEY_PARAMS = ("temperature", "top_p", "max_tokens", "max_completion_tokens", "stop", "tools", "tool_choice",
              "response_format", "seed", "n", "presence_penalty", "frequency_penalty")


class Embedder:
    def __init__(self, model: str):
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model)

    def embed(self, texts: list[str]) -> np.ndarray:
        v = np.asarray(list(self.model.embed(texts)), dtype=np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)


class EmbedBatcher:
    """Dynamic batching for the embedder.

    Requests that arrive while a batch is being embedded, or within `max_wait` of the first one, are embedded
    together in one ONNX call. Under load the batch size grows with concurrency on its own; at low traffic a
    request waits at most `max_wait`.
    """

    def __init__(self, embedder: Embedder, max_batch: int = 16, max_wait: float = 0.002):
        self.embedder = embedder
        self.max_batch = max_batch
        self.max_wait = max_wait
        self._queue: list[tuple[str, asyncio.Future]] = []
        self._task: asyncio.Task | None = None
        self.batches = 0
        self.items = 0

    async def embed(self, text: str) -> np.ndarray:
        fut = asyncio.get_running_loop().create_future()
        self._queue.append((text, fut))
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._drain())
        return await fut

    async def _drain(self) -> None:
        while self._queue:
            if len(self._queue) < self.max_batch:
                await asyncio.sleep(self.max_wait)
            batch, self._queue = self._queue[: self.max_batch], self._queue[self.max_batch:]
            try:
                vecs = await asyncio.to_thread(self.embedder.embed, [t for t, _ in batch])
            except Exception as e:  # noqa: BLE001 - hand the failure to every waiter
                for _, f in batch:
                    if not f.done():
                        f.set_exception(e)
                continue
            self.batches += 1
            self.items += len(batch)
            for (_, f), v in zip(batch, vecs):
                if not f.done():
                    f.set_result(v)


def _digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass
class CacheKeys:
    exact: str
    partition: str
    query: str
    vec: np.ndarray | None = None  # set by a semantic lookup, reused when the response is stored


def cache_keys(model: str, req: dict) -> CacheKeys | None:
    msgs = req["messages"]
    if not msgs or msgs[-1].get("role") != "user":
        return None
    params = {k: req[k] for k in KEY_PARAMS if req.get(k) is not None}
    context = [{"role": m["role"], "content": message_text(m.get("content"))} for m in msgs[:-1]]
    query = message_text(msgs[-1].get("content"))
    partition = _digest({"model": model, "context": context, "params": params})
    exact = _digest({"partition": partition, "query": query})
    return CacheKeys(exact, partition, query)


@dataclass
class Entry:
    key: str
    partition: str
    query: str
    body: dict
    cost_usd: float
    created: float


@dataclass
class Hit:
    body: dict
    kind: str  # "exact" or "semantic"
    similarity: float
    cost_usd: float  # what the original call cost: the saving from this hit
    matched_query: str


class _Partition:
    """Entries sharing one context, with their query embeddings stacked for a single matrix-vector product."""

    def __init__(self) -> None:
        self.keys: list[str] = []
        self.vecs: list[np.ndarray] = []
        self._matrix: np.ndarray | None = None

    def add(self, key: str, vec: np.ndarray) -> None:
        self.keys.append(key)
        self.vecs.append(vec)
        self._matrix = None

    def remove(self, key: str) -> None:
        i = self.keys.index(key)
        del self.keys[i], self.vecs[i]
        self._matrix = None

    def candidates(self, q: np.ndarray, threshold: float, k: int = 5) -> list[tuple[str, float]]:
        """Up to `k` entries at or above `threshold`, most similar first."""
        if not self.keys:
            return []
        if self._matrix is None:
            self._matrix = np.stack(self.vecs)
        sims = self._matrix @ q
        top = np.argsort(-sims)[:k] if len(sims) > k else np.argsort(-sims)
        return [(self.keys[i], float(sims[i])) for i in top if sims[i] >= threshold]


class ResponseCache:
    def __init__(self, cfg: CacheConfig, store: Store | None, embedder: Embedder | None = None):
        self.cfg = cfg
        self.store = store
        self.embedder = embedder if embedder is not None or not cfg.semantic else Embedder(cfg.embed_model)
        self.batcher = EmbedBatcher(self.embedder) if self.embedder else None
        self.entries: OrderedDict[str, Entry] = OrderedDict()  # insertion order = age, for eviction
        self.partitions: dict[str, _Partition] = {}
        self.vecs: dict[str, np.ndarray] = {}
        self.guard_rejections = 0
        if store:
            self._load()

    def _load(self) -> None:
        for key, partition, query, emb, body, cost, created in self.store.cache_load(time.time() - self.cfg.ttl_seconds):
            vec = np.frombuffer(emb, dtype=np.float32) if emb is not None else None
            self._insert(Entry(key, partition, query, json.loads(body), cost, created), vec)

    def cacheable(self, req: dict, mode: str | None) -> bool:
        if not self.cfg.enabled or mode == "off":
            return False
        if mode == "force":
            return True
        # Sampling at temperature > 0 is a request for variety; a cache would defeat it. OpenAI's default is 1.
        return req.get("temperature", 1.0) <= self.cfg.max_temperature and req.get("n", 1) == 1

    async def get(self, keys: CacheKeys, semantic: bool = True) -> Hit | None:
        now = time.time()
        e = self.entries.get(keys.exact)
        if e and now - e.created < self.cfg.ttl_seconds:
            return Hit(e.body, "exact", 1.0, e.cost_usd, e.query)
        if not (semantic and self.cfg.semantic and self.embedder):
            return None
        part = self.partitions.get(keys.partition)
        if not part or not part.keys:
            return None
        q = keys.vec = await self.batcher.embed(keys.query)
        for key, sim in part.candidates(q, self.cfg.threshold):
            e = self.entries[key]
            if now - e.created >= self.cfg.ttl_seconds:
                continue
            if self.cfg.guard and not guard.check(e.query, keys.query)[0]:
                self.guard_rejections += 1
                continue
            return Hit(e.body, "semantic", sim, e.cost_usd, e.query)
        return None

    async def put(self, keys: CacheKeys, body: dict, cost_usd: float) -> None:
        vec = keys.vec
        if vec is None and self.cfg.semantic and self.batcher:
            vec = await self.batcher.embed(keys.query)
        entry = Entry(keys.exact, keys.partition, keys.query, body, cost_usd, time.time())
        evicted = self._insert(entry, vec)
        if self.store:
            blob = vec.astype(np.float32).tobytes() if vec is not None else None
            await asyncio.to_thread(self.store.cache_put, entry.key, entry.partition, entry.query, blob,
                                    json.dumps(body), cost_usd)
            if evicted:
                await asyncio.to_thread(self.store.cache_delete, evicted)

    def _insert(self, entry: Entry, vec: np.ndarray | None) -> list[str]:
        if entry.key in self.entries:
            self._remove(entry.key)
        self.entries[entry.key] = entry
        if vec is not None:
            self.vecs[entry.key] = vec
            self.partitions.setdefault(entry.partition, _Partition()).add(entry.key, vec)
        evicted = []
        while len(self.entries) > self.cfg.max_entries:
            oldest = next(iter(self.entries))
            self._remove(oldest)
            evicted.append(oldest)
        return evicted

    def _remove(self, key: str) -> None:
        e = self.entries.pop(key)
        if self.vecs.pop(key, None) is not None:
            part = self.partitions[e.partition]
            part.remove(key)
            if not part.keys:
                del self.partitions[e.partition]

    def stats(self) -> dict:
        return {"entries": len(self.entries), "partitions": len(self.partitions),
                "semantic": bool(self.cfg.semantic and self.embedder), "threshold": self.cfg.threshold,
                "guard": self.cfg.guard, "guard_rejections": self.guard_rejections,
                "embed_batches": self.batcher.batches if self.batcher else 0,
                "mean_embed_batch": round(self.batcher.items / self.batcher.batches, 2)
                if self.batcher and self.batcher.batches else None}
