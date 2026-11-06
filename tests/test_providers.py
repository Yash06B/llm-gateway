import asyncio
import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from gateway.config import ProviderConfig
from gateway.providers import AnthropicProvider, OpenAIProvider, UpstreamError


def with_upstream(handler, fn):
    """Run `fn(base_url)` against a throwaway aiohttp server whose every POST goes to `handler`."""

    async def main():
        app = web.Application()
        app.router.add_post("/{tail:.*}", handler)
        server = TestServer(app)
        await server.start_server()
        try:
            return await fn(str(server.make_url("")).rstrip("/"))
        finally:
            await server.close()

    return asyncio.run(main())


def anthropic(base_url):
    p = AnthropicProvider(ProviderConfig("anthropic", "anthropic", base_url=base_url))
    p.available = True
    return p


HI = {"messages": [{"role": "user", "content": "hi"}]}


def test_anthropic_request_translation():
    body = AnthropicProvider.translate_request("claude-x", {
        "messages": [
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": [{"type": "text", "text": "Hi"}]},
            {"role": "user", "content": "Still there?"},
            {"role": "assistant", "content": "Yes."},
            {"role": "user", "content": "Good."},
        ],
        "temperature": 0, "stop": "END",
    })
    assert body["system"] == "Be terse."
    assert body["messages"] == [
        {"role": "user", "content": "Hi\n\nStill there?"},
        {"role": "assistant", "content": "Yes."},
        {"role": "user", "content": "Good."},
    ]
    assert body["max_tokens"] == 1024 and body["temperature"] == 0 and body["stop_sequences"] == ["END"]


def test_anthropic_complete_maps_response():
    async def handler(request):
        assert request.path == "/v1/messages" and request.headers["anthropic-version"]
        return web.json_response({
            "content": [{"type": "text", "text": "Hello there"}], "stop_reason": "max_tokens",
            "usage": {"input_tokens": 12, "output_tokens": 3},
        })

    async def run(url):
        p = anthropic(url)
        try:
            return await p.complete("claude-x", HI)
        finally:
            await p.aclose()

    out = with_upstream(handler, run)
    assert out["choices"][0]["message"]["content"] == "Hello there"
    assert out["choices"][0]["finish_reason"] == "length"
    assert out["usage"] == {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}


def test_anthropic_stream_maps_events():
    events = [
        {"type": "message_start", "message": {"usage": {"input_tokens": 7}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hel"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "lo"}},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 2}},
        {"type": "message_stop"},
    ]
    sse = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)

    async def handler(request):
        assert (await request.json())["stream"] is True
        return web.Response(text=sse, content_type="text/event-stream")

    async def run(url):
        p = anthropic(url)
        try:
            return [c async for c in p.stream("claude-x", HI)]
        finally:
            await p.aclose()

    chunks = with_upstream(handler, run)
    assert "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"]) == "Hello"
    assert [c["choices"][0]["finish_reason"] for c in chunks if c["choices"]][-1] == "stop"
    assert chunks[-1]["usage"] == {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9}


def test_openai_stream_relays_chunks_until_done():
    chunks = [{"choices": [{"index": 0, "delta": {"content": w}}]} for w in ("a", "b")]
    sse = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"

    async def handler(request):
        body = await request.json()
        assert body["stream"] is True and body["stream_options"] == {"include_usage": True}
        assert request.headers["authorization"] == "Bearer sk-test"
        return web.Response(text=sse, content_type="text/event-stream")

    async def run(url):
        import os

        os.environ["TEST_OPENAI_KEY"] = "sk-test"
        p = OpenAIProvider(ProviderConfig("up", "openai", base_url=url + "/v1", api_key_env="TEST_OPENAI_KEY"))
        try:
            return [c async for c in p.stream("m", HI)]
        finally:
            await p.aclose()

    assert with_upstream(handler, run) == chunks


@pytest.mark.parametrize("status,retryable", [(429, True), (500, True), (503, True), (400, False), (401, False)])
def test_openai_error_classification(status, retryable):
    async def handler(request):
        return web.json_response({"error": "x"}, status=status)

    async def run(url):
        p = OpenAIProvider(ProviderConfig("up", "openai", base_url=url + "/v1"))
        try:
            with pytest.raises(UpstreamError) as ei:
                await p.complete("m", HI)
            return ei.value
        finally:
            await p.aclose()

    err = with_upstream(handler, run)
    assert err.status == status and err.retryable is retryable


def test_timeout_and_connection_errors_are_retryable():
    async def slow(request):
        await asyncio.sleep(2)
        return web.json_response({})

    async def run(url):
        p = OpenAIProvider(ProviderConfig("up", "openai", base_url=url + "/v1", timeout=0.2))
        try:
            with pytest.raises(UpstreamError) as ei:
                await p.complete("m", HI)
            return ei.value
        finally:
            await p.aclose()

    err = with_upstream(slow, run)
    assert err.retryable and "timeout" in str(err)

    async def refused():
        p = OpenAIProvider(ProviderConfig("up", "openai", base_url="http://127.0.0.1:9/v1", timeout=1))
        try:
            with pytest.raises(UpstreamError) as ei:
                await p.complete("m", HI)
            return ei.value
        finally:
            await p.aclose()

    assert asyncio.run(refused()).retryable


def test_provider_without_key_is_unavailable(monkeypatch):
    monkeypatch.delenv("NO_SUCH_KEY", raising=False)
    assert not OpenAIProvider(ProviderConfig("o", "openai", base_url="http://x", api_key_env="NO_SUCH_KEY")).available
    assert OpenAIProvider(ProviderConfig("o", "openai", base_url="http://x")).available
