import json

import pytest
from fastapi.testclient import TestClient

from gateway.app import Gateway, create_app
from gateway.providers import UpstreamError

ADMIN = "test-admin"


class BadRequestProvider:
    """Fails like an upstream rejecting a malformed request: not the upstream's fault, so no fallback."""

    name, available = "primary", True

    async def complete(self, model, req):
        raise UpstreamError("primary: HTTP 400: bad request", 400, retryable=False)

    async def stream(self, model, req):
        raise UpstreamError("primary: HTTP 400: bad request", 400, retryable=False)
        yield  # pragma: no cover

    async def aclose(self):
        pass


@pytest.fixture
def setup(make_cfg, fake_embedder):
    def make(rpm=1000, tpm=1_000_000, providers=None, **cfg_overrides):
        cfg = make_cfg(**cfg_overrides)
        gw = Gateway(cfg, admin_key=ADMIN, providers=providers, embedder=fake_embedder)
        key, _ = gw.store.create_key("tester", rpm, tpm)
        client = TestClient(create_app(cfg, gw))
        client.headers["authorization"] = f"Bearer {key}"
        return client, gw

    return make


def chat(text, model="chat", **kw):
    return {"model": model, "temperature": 0, "messages": [{"role": "user", "content": text}], **kw}


def sse_events(resp):
    return [line[6:] for line in resp.text.splitlines() if line.startswith("data: ")]


def test_auth_validation_and_unknown_model(setup):
    client, _ = setup()
    with client:
        assert client.post("/v1/chat/completions", json=chat("hi"), headers={"authorization": "Bearer nope"}).status_code == 401
        assert client.post("/v1/chat/completions", json={"model": "chat"}).status_code == 400
        r = client.post("/v1/chat/completions", json=chat("hi", model="gpt-nope"))
        assert r.status_code == 404 and r.json()["error"]["type"] == "not_found_error"
        assert [m["id"] for m in client.get("/v1/models").json()["data"]] == ["chat", "solo"]


def test_completion_then_exact_then_semantic_hit(setup):
    client, gw = setup()
    with client:
        r = client.post("/v1/chat/completions", json=chat("What is the capital of France?"))
        assert r.status_code == 200
        assert r.headers["x-gateway-cache"] == "miss" and r.headers["x-gateway-target"] == "primary/p-1"
        assert r.json()["choices"][0]["message"]["content"].startswith("Mock answer to: What is the capital")

        r2 = client.post("/v1/chat/completions", json=chat("What is the capital of France?"))
        assert r2.headers["x-gateway-cache"] == "exact"
        assert r2.json()["choices"] == r.json()["choices"] and r2.json()["id"] != r.json()["id"]
        assert r2.json()["usage"]["total_tokens"] == 0

        r3 = client.post("/v1/chat/completions", json=chat("what is the capital of france"))
        assert r3.headers["x-gateway-cache"] == "semantic"

        r4 = client.post("/v1/chat/completions", json=chat("What is the capital of France?"),
                         headers={"x-gateway-cache": "off"})
        assert "x-gateway-cache" not in r4.headers
        assert gw.providers["primary"].calls == 2

        stats = client.get("/admin/stats", headers={"authorization": f"Bearer {ADMIN}"}).json()
        assert stats["cache_hits"] == {"exact": 1, "semantic": 1}
        assert stats["totals"]["saved_usd"] > 0 and stats["totals"]["cost_usd"] > 0


def test_sampled_requests_are_not_cached(setup):
    client, gw = setup()
    with client:
        for _ in range(2):
            r = client.post("/v1/chat/completions", json=chat("tell me a story", temperature=0.9))
            assert "x-gateway-cache" not in r.headers
        assert gw.providers["primary"].calls == 2


def test_fallback_then_circuit_opens(setup):
    client, gw = setup()
    gw.providers["primary"].fail_rate = 1.0
    with client:
        for i in range(3):
            r = client.post("/v1/chat/completions", json=chat(f"q{i}"))
            assert r.status_code == 200 and r.headers["x-gateway-target"] == "backup/b-1"
        # min_calls 2 at 50%: the third request skipped primary without calling it
        assert gw.providers["primary"].calls == 2
        assert gw.router.breakers["primary/p-1"].state == "open"
        assert r.headers["x-gateway-attempts"] == "1"

        r = client.post("/v1/chat/completions", json=chat("q9", model="solo"))
        assert r.status_code == 503 and int(r.headers["retry-after"]) > 0
        assert r.json()["error"]["attempts"][0]["error"] == "skipped: circuit open"


def test_all_targets_fail_returns_502_with_attempts(setup):
    client, gw = setup()
    gw.providers["primary"].fail_rate = gw.providers["backup"].fail_rate = 1.0
    with client:
        r = client.post("/v1/chat/completions", json=chat("hello"))
        assert r.status_code == 502
        assert [a["target"] for a in r.json()["error"]["attempts"]] == ["primary/p-1", "backup/b-1"]


def test_client_errors_are_not_retried(setup, make_cfg):
    from gateway.providers import MockProvider
    from gateway.config import ProviderConfig

    backup = MockProvider(ProviderConfig("backup", "mock"))
    client, gw = setup(providers={"primary": BadRequestProvider(), "backup": backup})
    with client:
        r = client.post("/v1/chat/completions", json=chat("hello"))
        assert r.status_code == 400 and backup.calls == 0
        assert gw.router.breakers["primary/p-1"].state == "closed"


def test_streaming_relay_usage_and_cached_replay(setup):
    client, gw = setup()
    with client:
        r = client.post("/v1/chat/completions", json=chat("Stream me a reply", stream=True))
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        events = sse_events(r)
        assert events[-1] == "[DONE]"
        chunks = [json.loads(e) for e in events[:-1]]
        assert all(c["choices"] for c in chunks), "usage-only chunk dropped when the client didn't ask for it"
        text = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks)
        assert text == "Mock answer to: Stream me a reply"

        r2 = client.post("/v1/chat/completions", json=chat("Stream me a reply", stream=True,
                                                             stream_options={"include_usage": True}))
        assert r2.headers["x-gateway-cache"] == "exact"
        chunks = [json.loads(e) for e in sse_events(r2)[:-1]]
        assert "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks if c["choices"]) == text
        assert chunks[-1]["usage"]["total_tokens"] == 0

        # a cached streaming answer also serves a non-streaming request
        r3 = client.post("/v1/chat/completions", json=chat("Stream me a reply"))
        assert r3.headers["x-gateway-cache"] == "exact" and r3.json()["choices"][0]["message"]["content"] == text
        assert gw.providers["primary"].calls == 1


def test_stream_fails_over_before_first_chunk(setup):
    client, gw = setup()
    gw.providers["primary"].fail_rate = 1.0
    with client:
        r = client.post("/v1/chat/completions", json=chat("hi", stream=True))
        assert r.status_code == 200 and r.headers["x-gateway-target"] == "backup/b-1"
        assert sse_events(r)[-1] == "[DONE]"


def test_rate_limit(setup):
    client, _ = setup(rpm=2)
    with client:
        assert client.post("/v1/chat/completions", json=chat("a")).status_code == 200
        r = client.post("/v1/chat/completions", json=chat("b"))
        assert r.status_code == 200 and r.headers["x-ratelimit-remaining-requests"] == "0"
        r = client.post("/v1/chat/completions", json=chat("c"))
        assert r.status_code == 429 and int(r.headers["retry-after"]) >= 1


def test_usage_accounting_and_admin(setup):
    client, gw = setup()
    admin = {"authorization": f"Bearer {ADMIN}"}
    with client:
        client.post("/v1/chat/completions", json=chat("count my tokens please"))
        client.post("/v1/chat/completions", json=chat("count my tokens please"))
        assert client.get("/admin/stats").status_code == 401

        rows = client.get("/admin/usage?group_by=cache", headers=admin).json()["rows"]
        by = {r["cache"]: r for r in rows}
        assert by["miss"]["requests"] == 1 and by["exact"]["requests"] == 1
        miss = by["miss"]
        # primary/p-1 costs $1/M input and $2/M output
        assert miss["cost_usd"] == pytest.approx((miss["prompt_tokens"] * 1 + miss["completion_tokens"] * 2) / 1e6)
        assert by["exact"]["saved_usd"] == pytest.approx(miss["cost_usd"])

        r = client.post("/admin/keys", json={"name": "svc", "rpm": 5}, headers=admin)
        assert r.json()["key"].startswith("gw-") and r.json()["rpm"] == 5
        assert len(client.get("/admin/keys", headers=admin).json()["keys"]) == 2

        prom = client.get("/metrics", headers=admin).text
        assert 'gateway_requests_total{model="chat",target="primary/p-1",status="2xx",cache="miss"} 1' in prom
        assert "gateway_saved_usd_total" in prom


def test_direct_provider_addressing(setup):
    client, _ = setup()
    with client:
        r = client.post("/v1/chat/completions", json=chat("hi", model="backup/some-model"))
        assert r.status_code == 200 and r.headers["x-gateway-target"] == "backup/some-model"
