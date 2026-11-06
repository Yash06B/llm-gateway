# llm-gateway

**A self-hosted, OpenAI-compatible LLM gateway: fallback routing with circuit breakers, an exact + semantic response
cache, per-key rate limits, cost accounting and a live dashboard.** Point any OpenAI SDK at it and it routes to
Ollama, Anthropic, OpenAI or anything OpenAI-compatible.

**Headline:**
- **Overhead:** at 64 concurrent requests the gateway adds **5 ms p50 / 13 ms p95** and keeps **92% of direct
  throughput** (782 vs 853 req/s, single process).
- **Resilience:** with the primary provider hung, the circuit breaker cuts **p50 from 1,084 ms to 71 ms**; with it
  failing 30% of calls, fallback lifts success from **70% to 100%**.
- **Cache safety:** semantic caching **served wrong answers to 15% of near-miss prompts** ("Celsius to Fahrenheit" vs
  "Fahrenheit to Celsius" scores 0.997 similarity). A lexical guard cut that to **0% on a held-out set at the same
  65% paraphrase hit rate**.
- **Against a real model:** a repeated question to `llama3.2:3b` drops from **2.47 s to 1.8 ms**; a recognised
  paraphrase to **17 ms**.

Part of a series: claude-overlay (local-first desktop copilot, Claude + Ollama) ·
[askdocs](https://github.com/Yash06B/askdocs) (RAG with an eval harness) ·
[triage-agent](https://github.com/Yash06B/triage-agent) (MCP agent for issue triage) ·
[slm-finetune](https://github.com/Yash06B/slm-finetune) (QLoRA text-to-SQL, small model vs big) ·
**this repo** (serving infrastructure: the layer between apps and models).

## What it does

```
client (any OpenAI SDK) ──► /v1/chat/completions
                              │ auth (hashed API keys) ─► rate limit (RPM + TPM token buckets, per key)
                              │ cache lookup: exact hash ─► semantic (embed + cosine + lexical guard)
                              │ route: model alias ─► [target 1, target 2, ...] skipping open circuits
                              │ stream or not; fail over until the first byte
                              └ usage log (tokens, cost, latency, cache, attempts) ─► SQLite, /metrics, /dashboard
```

- **One API, many providers.** Clients ask for an alias like `fast` or `smart`; the config maps it to an ordered
  list of `provider/model` targets. OpenAI-compatible upstreams (OpenAI, Ollama, vLLM, Groq, OpenRouter) pass
  through; Anthropic is translated both ways, streaming included. Any model is also addressable directly, e.g.
  `ollama/gemma3:4b`.
- **Fallback + circuit breakers.** Timeouts, 429s and 5xx fail over to the next target; 4xx client errors don't
  (they'd fail everywhere). A breaker per target trips on failure *rate* over a rolling window, then probes once per
  cooldown. Streams fail over only before the first chunk; after that the client already has partial output.
- **Exact + semantic cache.** Deterministic requests (`temperature: 0`, or `x-gateway-cache: force`) are cached.
  The semantic tier matches the last user turn by embedding similarity, but only inside an identical context (model,
  system prompt, earlier turns, sampling params), and only if the lexical guard agrees. Cached answers replay as SSE
  for streaming clients. Responses carry `x-gateway-cache: miss|exact|semantic` and the similarity.
- **Rate limits.** Requests/min and tokens/min per key as token buckets. Tokens are charged with the real count when
  the response finishes, so a key can go into debt and wait, like OpenAI's and Anthropic's own limits. Standard
  `x-ratelimit-*` and `retry-after` headers.
- **Cost accounting.** Per-request tokens, cost (from a price table), and the cost a cache hit *saved*, in SQLite:
  `/admin/usage?group_by=key|model|target|cache`.
- **Observability.** Prometheus `/metrics`; `/admin/stats` with p50/p95/p99 for misses, hits, gateway overhead and
  TTFT, breaker states and cache stats; and a `/dashboard` page that polls it.

## Results

All numbers are from one laptop (Intel i5-13420H, 16 GB, Windows 11, Python 3.12) with the load generator on the
same machine, so treat absolute throughput as a floor. Raw output for everything: [`eval/results/`](eval/results/).

### 1. Overhead and throughput

A mock OpenAI-compatible upstream with 50 ms latency; 4 client processes × 16 concurrent requests; 20 s per run
([`bench/loadtest.py`](bench/loadtest.py), [`loadtest.json`](eval/results/loadtest.json)).

| Path | Throughput | p50 | p95 | p99 |
|---|---:|---:|---:|---:|
| Direct to upstream (baseline) | 853 req/s | 75.9 ms | 94.4 ms | 102.0 ms |
| Through gateway, cache off | 782 req/s | 80.9 ms | 107.4 ms | 127.2 ms |
| Through gateway, every request a cache miss | 304 req/s | 210.9 ms | 248.5 ms | 271.1 ms |
| Through gateway, every request an exact hit | 2,328 req/s | 27.8 ms | 33.5 ms | 36.9 ms |
| Through gateway, paraphrased prompts | 578 req/s | 154.9 ms | 175.7 ms | 203.3 ms |

- **Proxy overhead is 5 ms p50, 13 ms p95**: auth, rate limiting, routing, accounting and one extra HTTP hop.
- **The cache-miss path is bound by the embedder, not the gateway.** bge-small peaks at ~330 embeddings/s on this
  CPU ([`micro.json`](eval/results/micro.json)) and every miss needs one. Against a real model that's a good trade
  (a hit skips a multi-second generation), but if misses dominate, set `semantic = false` and keep exact caching.

### 2. Is semantic caching safe? Not on embeddings alone.

170 hand-written prompt pairs ([`eval/cache_pairs.py`](eval/cache_pairs.py)): paraphrases that *should* share an
answer, and hard near-misses that *must not* (swapped entity, number, direction, negation, task). Full tables:
[`eval/results/cache.md`](eval/results/cache.md).

The most similar near-misses, by `bge-small-en-v1.5` cosine similarity:

| Similarity | Prompt A | Prompt B |
|---:|---|---|
| 0.997 | How do I convert Celsius to Fahrenheit? | How do I convert Fahrenheit to Celsius? |
| 0.997 | What is the flight time from London to New York? | What is the flight time from New York to London? |
| 0.992 | Convert 10 kilometers to miles. | Convert 10 miles to kilometers. |
| 0.975 | Summarize Hamlet in two sentences. | Summarize Hamlet in two paragraphs. |
| 0.969 | Sort this list in ascending order: 5, 2, 9. | Sort this list in descending order: 5, 2, 9. |
| 0.955 | How do I reverse a list in Python? | How do I reverse a list in JavaScript? |

These score **higher than the median real paraphrase (0.96)**, so no threshold separates them. At 0.95, 63% of
paraphrases hit and 16% of near-misses are served the other prompt's answer; lowering the threshold for more hits
makes it worse fast (41% wrong at 0.90).

**The fix is a lexical guard on every semantic match** ([`src/gateway/guard.py`](src/gateway/guard.py)). It rejects
a match when the numbers differ, when the same words move across a "to"/"into" pivot (a direction swap), or when
the prompts are the same sentence with one or two words substituted in place ("encrypt" / "decrypt"). Spelling
variants ("pyhton") don't count as substitutions. There are no antonym or entity word lists in it.

To keep the result honest, 40 of the pairs were written *before* the guard and never used to tune it:

| | Paraphrase hit rate | Near-misses served a wrong answer |
|---|---:|---:|
| Dev set (130 pairs), threshold 0.95, no guard | 63.3% | 15.7% (11 / 70) |
| Dev set, threshold 0.94 + guard (default) | 68.3% | **0%** (0 / 70) |
| **Held-out (40 pairs), threshold 0.95, no guard** | 65.0% | 15.0% (3 / 20) |
| **Held-out, threshold 0.94 + guard (default)** | 65.0% | **0%** (0 / 20) |

Same hit rate, no wrong answers. The cost: a paraphrase that swaps one word for a synonym in place ("Who **wrote**
X" vs "Who is the **author** of X") is rejected too, since lexically it looks exactly like "encrypt" / "decrypt".
The held-out set also caught a real bug, left unfixed so the number stays honest: "Quick **one**: what is…" is read
as the number 1.

### 3. Resilience

Two mock upstreams (primary and backup, 50 ms each); 2 × 8 concurrent clients; 20 s per run
([`bench/chaos.py`](bench/chaos.py), [`chaos.json`](eval/results/chaos.json)).

| Scenario | Success | Throughput | p50 | p99 |
|---|---:|---:|---:|---:|
| Primary fails 30% of calls, no fallback | 69.9% | 167 ok/s | 65 ms | 84 ms |
| Primary fails 30% of calls, fallback to backup | **100%** | 201 ok/s | 67 ms | 143 ms |
| Primary hung (1 s timeout), fallback, no breaker | 100% | 15 ok/s | 1,084 ms | 1,115 ms |
| Primary hung (1 s timeout), fallback + breaker | 100% | **220 ok/s** | **71 ms** | **86 ms** |

In the flaky run the primary still served 70% of traffic (2,812 of 4,019 requests): the breaker correctly left a
mostly-working provider in rotation.

### 4. Against a real model

20 held-out questions to `llama3.2:3b` on Ollama, then the same 20 again, then their paraphrases, then near-miss
variants ([`bench/ollama_live.py`](bench/ollama_live.py), [`ollama_live.json`](eval/results/ollama_live.json)).

| Round | Served from cache | p50 latency |
|---|---:|---:|
| First ask | 0 / 20 | 2,471 ms |
| Same question again | 20 / 20 exact | **1.8 ms** |
| Paraphrased | 13 / 20 semantic | 17 ms (hits) |
| Near-miss variants | 0 / 2 (correct) | 1,970 ms |

## What broke along the way

The benchmarks changed the design four times.

1. **One httpx client caps a single-process proxy.** The first version used one `httpx.AsyncClient` for upstream
   calls. With just 16 requests in flight it added **40 ms p50**; at 32 it fell to 82 req/s with a 1.1 s p95.
   Profiling showed the gateway's own work at 0.2 ms; the time was inside the client. Measured in isolation
   ([`bench/micro.py`](bench/micro.py)), a single httpx client gets *slower* as concurrency rises, while aiohttp
   scales with it:

   | One client, 50 ms upstream | 16 in flight | 64 in flight | 128 in flight |
   |---|---:|---:|---:|
   | httpx `AsyncClient` | 192 req/s | 91 req/s (p99 3.8 s) | 77 req/s (p99 6.3 s) |
   | aiohttp `ClientSession` | 255 req/s | 874 req/s | 1,479 req/s |

   After switching the upstream clients (and the load generator) to aiohttp, the gateway ran 455 req/s at 32 in
   flight with 8 ms p50 overhead.

2. **Every cache miss embedded the prompt twice** (once to look up, once to store), one ONNX call per request.
   Reusing the lookup's vector, plus a dynamic batcher that groups concurrent embeddings into one call
   (`EmbedBatcher`), took the miss path from 128 to 265 req/s at 32 in flight. Batch size then turned out to matter
   in both directions on CPU: 155 texts/s at batch 1, 331 at 16, 258 at 64. Capping batches at 16 took the
   64-in-flight miss path from 266 to 304 req/s.

3. **A consecutive-failure circuit breaker pulled a healthy provider out of rotation.** With "open after 3
   failures in a row" and a 30%-flaky primary, concurrent requests produce runs of three constantly, and the
   primary served only 14% of traffic. Switching to a failure-rate window exposed a second bug: requests already in
   flight when the circuit opened kept resetting its cooldown, so it never closed (0.5% success without fallback).
   The breaker now trips at an 80% failure rate over the last 20 calls and ignores late results. Its job is to skip
   *dead* providers; fallback already absorbs flaky ones.

4. **Templated prompts are a semantic-cache trap.** The first load test used prompts like "Request 0-3-12:
   summarise the benefits of connection pooling", different only in the ID. They score above 0.95 with each other,
   so the "every request a miss" run was 98% semantic hits. In production that's every templated prompt
   ("Summarise ticket #4411") returning the first ticket's summary. That run is why section 2 exists. (Numbers in
   the prompt are now caught by the guard.)

## Quickstart

```bash
git clone https://github.com/Yash06B/llm-gateway.git && cd llm-gateway
python -m venv .venv && .venv/Scripts/activate      # macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
cp gateway.example.toml gateway.toml                # providers, model aliases, prices
gateway keys create my-app --rpm 120                # prints the key once; only its hash is stored
GATEWAY_ADMIN_KEY=change-me gateway serve           # http://127.0.0.1:8787, dashboard at /dashboard
```

Then use it from any OpenAI client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8787/v1", api_key="gw-...")
r = client.chat.completions.create(model="fast", temperature=0,
                                   messages=[{"role": "user", "content": "What is the capital of Australia? One word."}])
```

```bash
curl -i http://127.0.0.1:8787/v1/chat/completions -H "Authorization: Bearer gw-..." \
  -d '{"model": "fast", "temperature": 0, "messages": [{"role": "user", "content": "what is the capital city of australia? one word"}]}'
# x-gateway-cache: semantic
# x-gateway-similarity: 0.9587
```

Per-request cache control: `x-gateway-cache: off` (bypass), `exact` (no semantic match), `force` (cache even at
temperature > 0).

| Endpoint | Auth | |
|---|---|---|
| `POST /v1/chat/completions` | API key | OpenAI-compatible, streaming and non-streaming |
| `GET /v1/models` | API key | Configured aliases and their targets |
| `GET /dashboard` | admin key, entered in the page | Live view of `/admin/stats` |
| `GET /admin/stats` | admin | Latency percentiles, cache, breakers, providers |
| `GET /admin/usage?hours=24&group_by=key` | admin | Requests, tokens, cost, savings |
| `GET` / `POST /admin/keys` | admin | List / create API keys |
| `GET /metrics` | admin | Prometheus text format |

## Reproducing the results

```bash
pytest -q                                         # 50 tests, ~2 s, no network or model downloads
python -m eval.cache_eval                         # threshold x guard sweep -> eval/results/cache.md
python -m bench.micro                             # httpx vs aiohttp, embedding batch sizes
python -m bench.loadtest --duration 20 --procs 4 --concurrency 16
python -m bench.chaos --duration 20
python -m bench.ollama_live --model llama3.2:3b   # needs Ollama running
```

The benchmarks start their own mock upstreams and gateways on ports 9101–9301 and 8788–8795.

## Project layout

```
src/gateway/
  app.py            FastAPI app: chat completions, streaming relay, admin, metrics, dashboard
  router.py         Fallback routing and the circuit breaker
  providers.py      OpenAI-compatible, Anthropic (translated) and mock providers, over aiohttp
  cache.py          Exact + semantic cache, context partitioning, EmbedBatcher
  guard.py          Lexical check on semantic matches
  ratelimit.py      Per-key RPM/TPM token buckets
  store.py          SQLite: hashed keys, batched usage log, cache persistence
  metrics.py        Rolling percentiles, Prometheus output
  mock_upstream.py  Fake upstream with latency and failure injection
  web/dashboard.html
bench/              Load, chaos, micro and live-Ollama benchmarks
eval/               Cache pair dataset (dev + held-out), the eval, and every result file
tests/              Breaker, rate limits, cache keys/TTL/eviction/persistence, guard, providers, end-to-end app
```

## Limitations and next steps

- **Single process.** Rate-limit buckets, breaker state and the cache index live in memory, so several workers
  would each have their own. Scaling out means moving them to Redis (atomic Lua scripts for the buckets).
- **The guard is lexical and English-only.** It can't tell that "Who wrote X" and "Who is the author of X" are the
  same question, and the held-out set is only 40 pairs. The next experiment is a small cross-encoder trained for
  paraphrase detection, measured against the guard on the same pairs.
- **Embedding throughput caps the semantic tier on CPU** (~330/s here). A GPU, a smaller model, or skipping the
  semantic lookup for very long prompts would raise it.
- **Anthropic translation is text-only.** Tool calls and images pass through to OpenAI-compatible upstreams but
  aren't translated for Anthropic yet.
- **Client, gateway and upstreams shared one machine** in every benchmark. The relative numbers (overhead,
  before/after) are the point; absolute throughput on a server would be higher.

## License

MIT
