"""Per-key token-bucket rate limits on requests per minute and tokens per minute.

Token usage isn't known until the response finishes, so the TPM bucket is checked up front (it must not be in
debt) and charged afterwards with the real count. A single huge response can push a key into debt; it then waits
for the bucket to refill, the same behaviour as OpenAI's and Anthropic's own limits.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


class TokenBucket:
    def __init__(self, per_minute: float, clock=time.monotonic):
        self.capacity = float(per_minute)
        self.rate = per_minute / 60.0
        self.tokens = self.capacity
        self.clock = clock
        self.ts = clock()

    def _refill(self) -> None:
        now = self.clock()
        self.tokens = min(self.capacity, self.tokens + (now - self.ts) * self.rate)
        self.ts = now

    def try_take(self, n: float = 1.0) -> float:
        """Take `n` if available and return 0, else return the seconds until it would be."""
        self._refill()
        if self.tokens >= n:
            self.tokens -= n
            return 0.0
        return (n - self.tokens) / self.rate

    def charge(self, n: float) -> None:
        self._refill()
        self.tokens -= n

    def remaining(self) -> int:
        self._refill()
        return max(0, int(self.tokens))


@dataclass
class Decision:
    allowed: bool
    retry_after: float
    limit: str  # which limit refused: "requests" or "tokens"
    headers: dict


class RateLimiter:
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self._buckets: dict[int, tuple[TokenBucket, TokenBucket]] = {}

    def _get(self, key_id: int, rpm: int, tpm: int) -> tuple[TokenBucket, TokenBucket]:
        b = self._buckets.get(key_id)
        if b is None or b[0].capacity != rpm or b[1].capacity != tpm:
            b = self._buckets[key_id] = (TokenBucket(rpm, self.clock), TokenBucket(tpm, self.clock))
        return b

    def check(self, key_id: int, rpm: int, tpm: int) -> Decision:
        req_b, tok_b = self._get(key_id, rpm, tpm)
        tok_b._refill()
        if tok_b.tokens < 0:
            return Decision(False, -tok_b.tokens / tok_b.rate, "tokens", self._headers(rpm, tpm, req_b, tok_b))
        wait = req_b.try_take(1.0)
        return Decision(wait == 0, wait, "requests", self._headers(rpm, tpm, req_b, tok_b))

    def charge_tokens(self, key_id: int, rpm: int, tpm: int, n: int) -> None:
        self._get(key_id, rpm, tpm)[1].charge(n)

    @staticmethod
    def _headers(rpm: int, tpm: int, req_b: TokenBucket, tok_b: TokenBucket) -> dict:
        return {
            "x-ratelimit-limit-requests": str(rpm),
            "x-ratelimit-remaining-requests": str(req_b.remaining()),
            "x-ratelimit-limit-tokens": str(tpm),
            "x-ratelimit-remaining-tokens": str(tok_b.remaining()),
        }
