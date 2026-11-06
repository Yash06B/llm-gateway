import hashlib
import re

import numpy as np
import pytest

from gateway.config import parse_config


class FakeEmbedder:
    """Bag-of-words hashing embedder: deterministic, no model download, similar texts -> similar vectors."""

    def embed(self, texts):
        out = np.zeros((len(texts), 256), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % 256] += 1
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


BASE = {
    "providers": {
        "primary": {"type": "mock"},
        "backup": {"type": "mock"},
    },
    "models": [
        {"name": "chat", "targets": [{"provider": "primary", "model": "p-1"}, {"provider": "backup", "model": "b-1"}]},
        {"name": "solo", "targets": [{"provider": "primary", "model": "p-1"}]},
    ],
    "prices": {"primary/p-1": [1.0, 2.0], "backup/b-1": [10.0, 20.0]},
    "cache": {"threshold": 0.8, "semantic": True},
    "routing": {"window": 4, "min_calls": 2, "failure_rate": 0.5, "cooldown_seconds": 60},
}


@pytest.fixture
def make_cfg(tmp_path):
    def make(**overrides):
        raw = {**BASE, **overrides}
        raw["server"] = {"db_path": str(tmp_path / "gw.db")}
        return parse_config(raw)

    return make


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()
