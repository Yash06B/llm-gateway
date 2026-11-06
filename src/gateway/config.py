"""Gateway configuration, loaded from a TOML file. See `gateway.example.toml`."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field, fields
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib


@dataclass
class ProviderConfig:
    name: str
    type: str  # "openai" (any OpenAI-compatible API, incl. Ollama), "anthropic" or "mock"
    base_url: str = ""
    api_key_env: str = ""
    timeout: float = 120.0
    # mock provider only
    latency_ms: float = 0.0
    fail_rate: float = 0.0
    token_delay_ms: float = 0.0

    @property
    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env) if self.api_key_env else None


@dataclass
class Target:
    provider: str
    model: str

    @property
    def id(self) -> str:
        return f"{self.provider}/{self.model}"


@dataclass
class ModelRoute:
    name: str
    targets: list[Target]


@dataclass
class CacheConfig:
    enabled: bool = True
    semantic: bool = True
    threshold: float = 0.94
    guard: bool = True  # lexical check on semantic matches; see gateway/guard.py
    ttl_seconds: float = 86400.0
    max_entries: int = 50_000
    # Only requests at or below this temperature are cached; clients can force with `x-gateway-cache: force`.
    max_temperature: float = 0.0
    embed_model: str = "BAAI/bge-small-en-v1.5"


@dataclass
class RoutingConfig:
    window: int = 20  # recent calls per target that the circuit breaker looks at
    min_calls: int = 10  # don't judge a target on fewer calls than this
    failure_rate: float = 0.8  # open the circuit when this share of the window failed
    cooldown_seconds: float = 30.0  # how long a circuit stays open before a half-open probe
    breaker_enabled: bool = True


@dataclass
class LimitsConfig:
    default_rpm: int = 60
    default_tpm: int = 100_000


@dataclass
class Config:
    db_path: str = "gateway.db"
    providers: dict[str, ProviderConfig] = field(default_factory=dict)
    models: dict[str, ModelRoute] = field(default_factory=dict)
    prices: dict[str, tuple[float, float]] = field(default_factory=dict)  # target id -> USD per 1M (input, output)
    cache: CacheConfig = field(default_factory=CacheConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    limits: LimitsConfig = field(default_factory=LimitsConfig)

    def price(self, target_id: str) -> tuple[float, float]:
        return self.prices.get(target_id, (0.0, 0.0))

    def cost(self, target_id: str, prompt_tokens: int, completion_tokens: int) -> float:
        p_in, p_out = self.price(target_id)
        return (prompt_tokens * p_in + completion_tokens * p_out) / 1_000_000


def _build(cls, section: str, values: dict, **extra):
    """Construct a config dataclass, naming the section and the bad key instead of raising a bare TypeError."""
    allowed = {f.name for f in fields(cls)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"[{section}]: unknown setting(s) {', '.join(unknown)}; "
                         f"valid: {', '.join(sorted(allowed - set(extra)))}")
    return cls(**values, **extra)


def parse_config(raw: dict) -> Config:
    server = raw.get("server", {})
    providers = {name: _build(ProviderConfig, f"providers.{name}", p, name=name)
                 for name, p in raw.get("providers", {}).items()}
    models: dict[str, ModelRoute] = {}
    for m in raw.get("models", []):
        targets = [Target(**t) for t in m["targets"]]
        for t in targets:
            if t.provider not in providers:
                raise ValueError(f"model {m['name']!r} routes to unknown provider {t.provider!r}")
        models[m["name"]] = ModelRoute(name=m["name"], targets=targets)
    return Config(
        db_path=server.get("db_path", "gateway.db"),
        providers=providers,
        models=models,
        prices={k: (float(v[0]), float(v[1])) for k, v in raw.get("prices", {}).items()},
        cache=_build(CacheConfig, "cache", raw.get("cache", {})),
        routing=_build(RoutingConfig, "routing", raw.get("routing", {})),
        limits=_build(LimitsConfig, "limits", raw.get("limits", {})),
    )


def load_config(path: str | Path) -> Config:
    with open(path, "rb") as f:
        return parse_config(tomllib.load(f))
