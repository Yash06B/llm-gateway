"""Request bodies for the load generator, by name (functions must be importable by spawned client processes).

Prompts come from the cache eval set, so "distinct" prompts really are distinct questions. (Templated prompts that
differ only by an ID, like "Request 17: ...", score above 0.95 similarity with each other: a semantic cache would
treat them all as one question.)
"""

import random

from eval.cache_pairs import DIFFERENT, HELDOUT_DIFFERENT, HELDOUT_SAME, SAME

QUESTIONS = [a for _, a, _ in SAME + HELDOUT_SAME]  # 80 distinct questions
PARAPHRASES = [b for _, _, b in SAME + HELDOUT_SAME]  # a paraphrase of each
VOCAB = sorted({w.strip("?.,!'\"").lower() for _, a, b in SAME + DIFFERENT + HELDOUT_SAME + HELDOUT_DIFFERENT
                for w in (a + " " + b).split() if len(w) > 3})


def _req(model: str, text: str) -> dict:
    return {"model": model, "temperature": 0, "max_tokens": 64, "messages": [{"role": "user", "content": text}]}


def unique(worker: int, slot: int, i: int) -> dict:
    """Never repeats, and never semantically close: random word salad, so every request is a cache miss."""
    rng = random.Random(f"{worker}-{slot}-{i}")
    return _req("bench", " ".join(rng.sample(VOCAB, 10)))


def unique_direct(worker: int, slot: int, i: int) -> dict:
    return {**unique(worker, slot, i), "model": "mock-model"}


def repeated(worker: int, slot: int, i: int) -> dict:
    """The 80 questions on repeat: after warm-up, exact hits."""
    return _req("bench", QUESTIONS[(slot * 7 + i) % len(QUESTIONS)])


def paraphrased(worker: int, slot: int, i: int) -> dict:
    """Paraphrases of the warmed questions: semantic hits where the cache recognises them, upstream calls otherwise."""
    return _req("bench", PARAPHRASES[(slot * 7 + i) % len(PARAPHRASES)])


def chaos(worker: int, slot: int, i: int) -> dict:
    return _req("chaos", f"chaos {worker}-{slot}-{i}")


def chaos_solo(worker: int, slot: int, i: int) -> dict:
    return _req("solo", f"chaos {worker}-{slot}-{i}")
