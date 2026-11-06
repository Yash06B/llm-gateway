"""Upstream providers. Every provider speaks the OpenAI chat-completions shape to the rest of the gateway.

- `OpenAIProvider` covers any OpenAI-compatible API: OpenAI, Ollama (`/v1`), vLLM, Groq, OpenRouter, ...
- `AnthropicProvider` translates to and from the Anthropic Messages API.
- `MockProvider` is an in-process fake with configurable latency and failure rate, for tests and benchmarks.

Upstream HTTP uses aiohttp rather than httpx: under load, a single httpx AsyncClient serialises on its connection
pool and throughput collapses as concurrency grows (90 req/s at 64 in flight vs 914 for aiohttp on the same mock
upstream; see the README).
"""

from __future__ import annotations

import asyncio
import json
import random
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Protocol

import aiohttp

from .config import ProviderConfig


class UpstreamError(Exception):
    """A failed upstream call. `retryable` errors make the router try the next target."""

    def __init__(self, message: str, status: int | None = None, retryable: bool = True):
        super().__init__(message)
        self.status = status
        self.retryable = retryable


def _retryable_status(status: int) -> bool:
    return status in (408, 409, 429) or status >= 500


class Provider(Protocol):
    name: str
    available: bool

    async def complete(self, model: str, req: dict) -> dict: ...

    def stream(self, model: str, req: dict) -> AsyncIterator[dict]: ...

    async def aclose(self) -> None: ...


def new_id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex[:24]


def make_completion(model: str, content: str, finish_reason: str, prompt_tokens: int, completion_tokens: int) -> dict:
    return {
        "id": new_id(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish_reason}
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def make_chunk(cid: str, model: str, delta: dict, finish_reason: str | None = None) -> dict:
    return {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def make_usage_chunk(cid: str, model: str, prompt_tokens: int, completion_tokens: int) -> dict:
    return {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def message_text(content) -> str:
    """Flatten OpenAI message content (a string or a list of parts) to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class _HTTPProvider:
    """Shared aiohttp plumbing. The session is created lazily because it must belong to the running event loop."""

    retryable_statuses: tuple[int, ...] = ()

    def __init__(self, cfg: ProviderConfig, base_url: str, headers: dict):
        self.name = cfg.name
        self.base_url = base_url.rstrip("/")
        self._headers = headers
        # sock_read bounds the wait for each read rather than the whole response, so long streams aren't cut off
        self._timeout = aiohttp.ClientTimeout(total=None, sock_connect=min(cfg.timeout, 5.0), sock_read=cfg.timeout)
        self._session: aiohttp.ClientSession | None = None

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers=self._headers, timeout=self._timeout,
                connector=aiohttp.TCPConnector(limit=1024, ttl_dns_cache=300),
            )
        return self._session

    @asynccontextmanager
    async def _post(self, path: str, body: dict):
        try:
            async with self._get_session().post(self.base_url + path, json=body) as r:
                if r.status >= 400:
                    text = await r.text(errors="replace")
                    raise UpstreamError(f"{self.name}: HTTP {r.status}: {text[:300]}", r.status,
                                        _retryable_status(r.status) or r.status in self.retryable_statuses)
                yield r
        except asyncio.TimeoutError as e:
            raise UpstreamError(f"{self.name}: timeout") from e
        except aiohttp.ClientError as e:
            raise UpstreamError(f"{self.name}: {type(e).__name__}: {e}") from e

    @staticmethod
    async def _sse(r: aiohttp.ClientResponse) -> AsyncIterator[dict | str]:
        async for raw in r.content:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            yield data if data == "[DONE]" else json.loads(data)

    async def aclose(self) -> None:
        if self._session is not None:
            await self._session.close()


class OpenAIProvider(_HTTPProvider):
    def __init__(self, cfg: ProviderConfig):
        super().__init__(cfg, cfg.base_url, {"Authorization": f"Bearer {cfg.api_key}"} if cfg.api_key else {})
        # A provider that needs a key it doesn't have is skipped by the router instead of failing every call.
        self.available = not cfg.api_key_env or bool(cfg.api_key)

    async def complete(self, model: str, req: dict) -> dict:
        body = {**req, "model": model, "stream": False}
        body.pop("stream_options", None)
        async with self._post("/chat/completions", body) as r:
            try:
                return await r.json(content_type=None)
            except ValueError as e:
                raise UpstreamError(f"{self.name}: invalid JSON from upstream") from e

    async def stream(self, model: str, req: dict) -> AsyncIterator[dict]:
        body = {**req, "model": model, "stream": True, "stream_options": {"include_usage": True}}
        async with self._post("/chat/completions", body) as r:  # also maps errors raised mid-stream
            async for event in self._sse(r):
                if event == "[DONE]":
                    return
                yield event


_STOP_REASONS = {"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length", "tool_use": "tool_calls"}


class AnthropicProvider(_HTTPProvider):
    """Translates OpenAI chat requests to the Anthropic Messages API (text content only)."""

    retryable_statuses = (529,)  # overloaded

    def __init__(self, cfg: ProviderConfig):
        key = cfg.api_key
        super().__init__(cfg, cfg.base_url or "https://api.anthropic.com",
                         {"x-api-key": key or "", "anthropic-version": "2023-06-01"})
        self.available = bool(key)

    @staticmethod
    def translate_request(model: str, req: dict) -> dict:
        system = "\n\n".join(message_text(m.get("content")) for m in req["messages"] if m["role"] == "system")
        messages = []
        for m in req["messages"]:
            if m["role"] == "system":
                continue
            role = "assistant" if m["role"] == "assistant" else "user"
            text = message_text(m.get("content"))
            if messages and messages[-1]["role"] == role:  # Anthropic requires alternating turns
                messages[-1]["content"] += "\n\n" + text
            else:
                messages.append({"role": role, "content": text})
        body = {"model": model, "messages": messages,
                "max_tokens": req.get("max_tokens") or req.get("max_completion_tokens") or 1024}
        if system:
            body["system"] = system
        for k in ("temperature", "top_p"):
            if req.get(k) is not None:
                body[k] = req[k]
        if req.get("stop"):
            body["stop_sequences"] = [req["stop"]] if isinstance(req["stop"], str) else req["stop"]
        return body

    async def complete(self, model: str, req: dict) -> dict:
        async with self._post("/v1/messages", self.translate_request(model, req)) as r:
            data = await r.json(content_type=None)
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        usage = data.get("usage", {})
        return make_completion(model, text, _STOP_REASONS.get(data.get("stop_reason"), "stop"),
                               usage.get("input_tokens", 0), usage.get("output_tokens", 0))

    async def stream(self, model: str, req: dict) -> AsyncIterator[dict]:
        body = {**self.translate_request(model, req), "stream": True}
        cid, prompt_tokens, completion_tokens = new_id(), 0, 0
        async with self._post("/v1/messages", body) as r:
            yield make_chunk(cid, model, {"role": "assistant", "content": ""})
            async for ev in self._sse(r):
                kind = ev.get("type")
                if kind == "message_start":
                    prompt_tokens = ev["message"].get("usage", {}).get("input_tokens", 0)
                elif kind == "content_block_delta" and ev["delta"].get("type") == "text_delta":
                    yield make_chunk(cid, model, {"content": ev["delta"]["text"]})
                elif kind == "message_delta":
                    completion_tokens = ev.get("usage", {}).get("output_tokens", completion_tokens)
                    reason = _STOP_REASONS.get(ev.get("delta", {}).get("stop_reason"), "stop")
                    yield make_chunk(cid, model, {}, reason)
                elif kind == "error":
                    raise UpstreamError(f"{self.name}: stream error: {ev.get('error')}")
        yield make_usage_chunk(cid, model, prompt_tokens, completion_tokens)


class MockProvider:
    """Deterministic fake upstream. Also served over HTTP by `gateway mock-upstream` for load tests."""

    def __init__(self, cfg: ProviderConfig, seed: int | None = None):
        self.name = cfg.name
        self.available = True
        self.latency = cfg.latency_ms / 1000
        self.fail_rate = cfg.fail_rate
        self.token_delay = cfg.token_delay_ms / 1000
        self._rng = random.Random(seed)
        self.calls = 0

    def _answer(self, req: dict) -> tuple[str, int]:
        last = next((message_text(m.get("content")) for m in reversed(req["messages"]) if m["role"] == "user"), "")
        prompt_tokens = sum(estimate_tokens(message_text(m.get("content"))) for m in req["messages"])
        return f"Mock answer to: {last[:200]}", prompt_tokens

    async def _maybe_fail(self) -> None:
        self.calls += 1
        if self.latency:
            await asyncio.sleep(self.latency)
        if self.fail_rate and self._rng.random() < self.fail_rate:
            raise UpstreamError(f"{self.name}: injected failure", 503)

    async def complete(self, model: str, req: dict) -> dict:
        await self._maybe_fail()
        text, prompt_tokens = self._answer(req)
        return make_completion(model, text, "stop", prompt_tokens, estimate_tokens(text))

    async def stream(self, model: str, req: dict) -> AsyncIterator[dict]:
        await self._maybe_fail()
        text, prompt_tokens = self._answer(req)
        cid = new_id()
        yield make_chunk(cid, model, {"role": "assistant", "content": ""})
        for i, word in enumerate(text.split(" ")):
            if self.token_delay:
                await asyncio.sleep(self.token_delay)
            yield make_chunk(cid, model, {"content": word if i == 0 else " " + word})
        yield make_chunk(cid, model, {}, "stop")
        yield make_usage_chunk(cid, model, prompt_tokens, estimate_tokens(text))

    async def aclose(self) -> None:
        pass


def make_provider(cfg: ProviderConfig) -> Provider:
    if cfg.type == "openai":
        return OpenAIProvider(cfg)
    if cfg.type == "anthropic":
        return AnthropicProvider(cfg)
    if cfg.type == "mock":
        return MockProvider(cfg)
    raise ValueError(f"provider {cfg.name!r}: unknown type {cfg.type!r}")
