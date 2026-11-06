import asyncio

import pytest

from gateway.cache import ResponseCache, cache_keys
from gateway.config import CacheConfig
from gateway.ratelimit import RateLimiter, TokenBucket
from gateway.router import Breaker


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_breaker_opens_on_failure_rate_and_probes_after_cooldown():
    clock = Clock()
    b = Breaker(window=10, min_calls=4, failure_rate=0.8, cooldown=10, clock=clock)
    for _ in range(3):
        b.record_failure()
    assert b.allow() and b.state == "closed", "not judged before min_calls"
    b.record_failure()
    assert b.state == "open" and not b.allow()
    clock.t = 9.9
    assert not b.allow()
    clock.t = 10.0
    assert b.allow() and b.state == "half_open"
    assert not b.allow(), "only one probe at a time while half-open"
    b.record_failure()
    assert b.state == "open", "a failed probe re-opens the circuit"
    clock.t = 20.0
    assert b.allow()
    b.record_success()
    assert b.state == "closed" and b.failures == 0, "a good probe closes it with a fresh window"


def test_late_results_dont_extend_an_open_circuit():
    clock = Clock()
    b = Breaker(window=10, min_calls=2, failure_rate=0.8, cooldown=10, clock=clock)
    b.record_failure(), b.record_failure()
    assert b.state == "open"
    clock.t = 9
    b.record_failure()  # a request that was in flight when the circuit opened
    clock.t = 10
    assert b.allow(), "cooldown still counts from when it opened"


def test_breaker_tolerates_a_flaky_but_mostly_healthy_target():
    b = Breaker(window=20, min_calls=10, failure_rate=0.8)
    pattern = [True, True, True, False, False, False, False, False, False, False]  # 30% failures, in runs of 3
    for failed in pattern * 5:
        b.record_failure() if failed else b.record_success()
        assert b.state == "closed"


def test_token_bucket_refills_over_time():
    clock = Clock()
    tb = TokenBucket(60, clock)  # 1 per second
    for _ in range(60):
        assert tb.try_take() == 0
    assert tb.try_take() == pytest.approx(1.0)
    clock.t = 0.5
    assert tb.try_take() == pytest.approx(0.5)
    clock.t = 1.0
    assert tb.try_take() == 0


def test_rate_limiter_rpm_and_token_debt():
    clock = Clock()
    rl = RateLimiter(clock)
    assert rl.check(1, rpm=2, tpm=600).allowed
    assert rl.check(1, rpm=2, tpm=600).allowed
    d = rl.check(1, rpm=2, tpm=600)
    assert not d.allowed and d.limit == "requests" and d.retry_after == pytest.approx(30)
    assert rl.check(2, rpm=2, tpm=600).allowed, "limits are per key"

    clock.t = 60
    rl.charge_tokens(1, 2, 600, 900)  # 300 tokens of debt at 10 tokens/s
    d = rl.check(1, rpm=2, tpm=600)
    assert not d.allowed and d.limit == "tokens" and d.retry_after == pytest.approx(30)
    clock.t = 90
    assert rl.check(1, rpm=2, tpm=600).allowed


def req(text, system="be brief", temperature=0, **kw):
    return {"model": "chat", "temperature": temperature,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": text}], **kw}


BODY = {"choices": [{"message": {"role": "assistant", "content": "Paris"}, "finish_reason": "stop"}]}


def test_cache_keys_separate_context_from_query():
    a = cache_keys("chat", req("What is the capital of France?"))
    b = cache_keys("chat", req("what's the capital of france"))
    c = cache_keys("chat", req("What is the capital of France?", system="answer in French"))
    d = cache_keys("chat", req("What is the capital of France?", max_tokens=5))
    assert a.partition == b.partition and a.exact != b.exact
    assert c.partition != a.partition, "a different system prompt is a different context"
    assert d.partition != a.partition, "params that change the answer are part of the key"
    assert cache_keys("chat", {"messages": [{"role": "assistant", "content": "hi"}]}) is None
    stream = cache_keys("chat", req("What is the capital of France?", stream=True, user="u1"))
    assert stream.exact == a.exact, "transport-only fields don't change the key"


def test_cacheable_policy(fake_embedder):
    cache = ResponseCache(CacheConfig(), None, fake_embedder)
    assert cache.cacheable(req("x"), None)
    assert not cache.cacheable(req("x", temperature=0.7), None)
    assert not cache.cacheable({"messages": []}, None), "OpenAI's default temperature is 1"
    assert cache.cacheable(req("x", temperature=0.7), "force")
    assert not cache.cacheable(req("x"), "off")


def test_exact_and_semantic_hits(fake_embedder):
    cache = ResponseCache(CacheConfig(threshold=0.8), None, fake_embedder)
    k1 = cache_keys("chat", req("What is the capital of France?"))
    asyncio.run(cache.put(k1, BODY, 0.01))

    hit = asyncio.run(cache.get(k1))
    assert hit.kind == "exact" and hit.cost_usd == 0.01

    hit = asyncio.run(cache.get(cache_keys("chat", req("what is the capital of france"))))
    assert hit and hit.kind == "semantic" and hit.similarity >= 0.8

    assert asyncio.run(cache.get(cache_keys("chat", req("How do I sort a list in Python?")))) is None
    other_ctx = cache_keys("chat", req("What is the capital of France?", system="answer in French"))
    assert asyncio.run(cache.get(other_ctx)) is None
    assert asyncio.run(cache.get(cache_keys("chat", req("what is the capital of france")), semantic=False)) is None


def test_ttl_and_eviction(fake_embedder):
    cache = ResponseCache(CacheConfig(max_entries=2, ttl_seconds=1000), None, fake_embedder)
    keys = [cache_keys("chat", req(q)) for q in ("alpha one", "beta two", "gamma three")]
    for k in keys:
        asyncio.run(cache.put(k, BODY, 0))
    assert len(cache.entries) == 2
    assert asyncio.run(cache.get(keys[0], semantic=False)) is None, "oldest entry evicted"
    assert asyncio.run(cache.get(keys[2]))
    cache.entries[keys[2].exact].created -= 2000
    assert asyncio.run(cache.get(keys[2])) is None, "expired"


def test_cache_persists_across_restarts(tmp_path, fake_embedder):
    from gateway.store import Store

    store = Store(str(tmp_path / "c.db"))
    cache = ResponseCache(CacheConfig(threshold=0.8), store, fake_embedder)
    asyncio.run(cache.put(cache_keys("chat", req("What is the capital of France?")), BODY, 0.02))
    store.close()

    store = Store(str(tmp_path / "c.db"))
    cache = ResponseCache(CacheConfig(threshold=0.8), store, fake_embedder)
    hit = asyncio.run(cache.get(cache_keys("chat", req("what is the capital of france"))))
    assert hit and hit.kind == "semantic" and hit.body == BODY and hit.cost_usd == 0.02


def test_embed_batcher_groups_concurrent_requests(fake_embedder):
    from gateway.cache import EmbedBatcher

    calls = []

    class Counting:
        def embed(self, texts):
            calls.append(len(texts))
            return fake_embedder.embed(texts)

    batcher = EmbedBatcher(Counting(), max_batch=8)

    async def run():
        return await asyncio.gather(*(batcher.embed(f"text number {i}") for i in range(20)))

    vecs = asyncio.run(run())
    assert len(vecs) == 20 and sum(calls) == 20
    assert calls == [8, 8, 4], "20 concurrent requests become 3 ONNX calls"
    assert (vecs[3] == fake_embedder.embed(["text number 3"])[0]).all(), "results go back to the right caller"


def test_miss_embeds_once(fake_embedder):
    calls = []

    class Counting:
        def embed(self, texts):
            calls.extend(texts)
            return fake_embedder.embed(texts)

    cache = ResponseCache(CacheConfig(threshold=0.99), None, Counting())
    asyncio.run(cache.put(cache_keys("chat", req("first question here")), BODY, 0))
    keys = cache_keys("chat", req("a totally unrelated prompt"))
    assert asyncio.run(cache.get(keys)) is None
    asyncio.run(cache.put(keys, BODY, 0))
    assert calls.count("a totally unrelated prompt") == 1, "the lookup's vector is reused on store"


def test_unknown_config_key_is_a_clear_error():
    from gateway.config import parse_config

    with pytest.raises(ValueError, match=r"\[routing\]: unknown setting\(s\) failure_threshold"):
        parse_config({"routing": {"failure_threshold": 3}})
