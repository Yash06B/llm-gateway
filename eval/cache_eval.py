"""How safe is semantic caching? Sweep the similarity threshold over labeled prompt pairs, with and without the
lexical guard (`gateway/guard.py`).

Pairwise, for each threshold:
  - paraphrase hit rate on SAME pairs: answers served from cache (the saving)
  - wrong-answer rate on DIFFERENT pairs: a near-miss served someone else's answer (the bug)

Whole cache: store every first prompt, query every second prompt against all of them (top-5 candidates, guard
applied in order, exactly like `ResponseCache.get`), and count a hit as correct only if the cached prompt is the
query's paraphrase (or the identical prompt).

Two datasets: `dev` (used while building the guard) and `heldout` (written before the guard, never used to tune it).

    python -m eval.cache_eval [--model BAAI/bge-small-en-v1.5]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from eval.cache_pairs import DIFFERENT, HELDOUT_DIFFERENT, HELDOUT_SAME, SAME
from gateway import guard
from gateway.cache import Embedder

OUT = Path(__file__).parent / "results"
THRESHOLDS = [0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.98]
DATASETS = {"dev": (SAME, DIFFERENT), "heldout": (HELDOUT_SAME, HELDOUT_DIFFERENT)}


def evaluate(emb: Embedder, same_pairs, diff_pairs) -> dict:
    pairs = [("same", c, a, b) for c, a, b in same_pairs] + [("different", c, a, b) for c, a, b in diff_pairs]
    va, vb = emb.embed([p[2] for p in pairs]), emb.embed([p[3] for p in pairs])
    sims = np.sum(va * vb, axis=1)
    same = np.array([p[0] == "same" for p in pairs])
    passes = np.array([guard.check(p[2], p[3])[0] for p in pairs])

    sweep = []
    for t in THRESHOLDS:
        row = {"threshold": t}
        for name, ok in (("off", np.ones_like(passes)), ("on", passes)):
            hit = (sims >= t) & ok
            row[name] = {"paraphrase_hit_rate": round(float(np.mean(hit[same])), 4),
                         "wrong_answer_rate": round(float(np.mean(hit[~same])), 4),
                         "wrong_answers": int(np.sum(hit[~same]))}
        sweep.append(row)

    stored = sorted({p[2] for p in pairs})
    full = vb @ emb.embed(stored).T
    # A cached answer is right for a query if both prompts are in the same equivalence class: linked by a SAME
    # pair or textually identical (several prompts appear in more than one pair).
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for label, _, a, b in pairs:
        find(a), find(b)
        if label == "same":
            parent[find(a)] = find(b)
    whole = []
    for t in THRESHOLDS:
        row = {"threshold": t}
        for name, use_guard in (("off", False), ("on", True)):
            served = correct = 0
            for qi, p in enumerate(pairs):
                for ci in np.argsort(-full[qi])[:5]:
                    if full[qi, ci] < t:
                        break
                    if use_guard and not guard.check(stored[ci], p[3])[0]:
                        continue
                    served += 1
                    correct += find(stored[ci]) == find(p[3])
                    break
            row[name] = {"served": round(served / len(pairs), 4), "correct": correct, "wrong": served - correct,
                         "wrong_rate": round((served - correct) / len(pairs), 4)}
        whole.append(row)

    cats: dict[str, list] = defaultdict(list)
    for (label, cat, _, _), s, g in zip(pairs, sims, passes):
        cats[f"{label}/{cat}"].append((float(s), bool(g)))
    by_cat = {k: {"n": len(v), "mean_sim": round(float(np.mean([s for s, _ in v])), 4),
                  "hit@0.95": round(float(np.mean([s >= 0.95 for s, _ in v])), 4),
                  "hit@0.90+guard": round(float(np.mean([s >= 0.90 and g for s, g in v])), 4)}
              for k, v in sorted(cats.items())}

    def listing(mask, order):
        return [{"sim": round(float(sims[i]), 4), "category": pairs[i][1], "a": pairs[i][2], "b": pairs[i][3],
                 "guard": guard.check(pairs[i][2], pairs[i][3])[1]} for i in order if mask[i]]

    return {
        "pairs": {"same": int(same.sum()), "different": int((~same).sum())},
        "sweep": sweep, "whole_cache": whole, "categories": by_cat,
        "highest_scoring_different": listing(~same, np.argsort(-sims))[:12],
        "lowest_scoring_same": listing(same, np.argsort(sims))[:6],
        "guard_misses": listing(~same & passes & (sims >= 0.90), np.argsort(-sims)),
        "guard_false_rejections": listing(same & ~passes, np.argsort(-sims)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    args = ap.parse_args()
    emb = Embedder(args.model)
    result = {"model": args.model, **{name: evaluate(emb, *data) for name, data in DATASETS.items()}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cache.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (OUT / "cache.md").write_text(render(result), encoding="utf-8")
    print(f"wrote {OUT / 'cache.md'}")


def _pct(x: float) -> str:
    return f"{x:.1%}"


def render(r: dict) -> str:
    out = [f"# Semantic cache eval: `{r['model']}`", "",
           "Generated by `python -m eval.cache_eval`. Pairs live in `eval/cache_pairs.py`.", "",
           "- **Paraphrase hit rate**: of the pairs that *should* share an answer, how many were served from cache.",
           "- **Wrong-answer rate**: of the hard near-misses (swapped entity, number, direction, negation), how many "
           "were served the other prompt's cached answer.", ""]
    for name in ("dev", "heldout"):
        d = r[name]
        title = {"dev": "Dev set (the guard was developed against this set)",
                 "heldout": "Held-out set (written before the guard, never used to tune it)"}[name]
        out += [f"## {title}", "", f"{d['pairs']['same']} paraphrase pairs, {d['pairs']['different']} near-miss pairs.",
                "", "### Pairwise", "",
                "| Threshold | Paraphrase hits | Wrong answers | Paraphrase hits + guard | Wrong answers + guard |",
                "|---:|---:|---:|---:|---:|"]
        for row in d["sweep"]:
            off, on = row["off"], row["on"]
            out.append(f"| {row['threshold']:.2f} | {_pct(off['paraphrase_hit_rate'])} | "
                       f"{_pct(off['wrong_answer_rate'])} ({off['wrong_answers']}) | "
                       f"{_pct(on['paraphrase_hit_rate'])} | {_pct(on['wrong_answer_rate'])} ({on['wrong_answers']}) |")
        out += ["", "### Whole cache", "",
                "Every first prompt cached; every second prompt looked up against all of them.", "",
                "| Threshold | Served | Correct | Wrong | Served + guard | Correct + guard | Wrong + guard |",
                "|---:|---:|---:|---:|---:|---:|---:|"]
        for row in d["whole_cache"]:
            off, on = row["off"], row["on"]
            out.append(f"| {row['threshold']:.2f} | {_pct(off['served'])} | {off['correct']} | {off['wrong']} | "
                       f"{_pct(on['served'])} | {on['correct']} | {on['wrong']} |")
        out += ["", "### By category", "", "| Label / category | Pairs | Mean sim | Hit @0.95 | Hit @0.90 + guard |",
                "|---|---:|---:|---:|---:|"]
        for k, v in d["categories"].items():
            out.append(f"| {k} | {v['n']} | {v['mean_sim']:.3f} | {v['hit@0.95']:.0%} | {v['hit@0.90+guard']:.0%} |")
        out += ["", "### Most similar near-misses", "", "| Sim | Category | A | B | Guard |", "|---:|---|---|---|---|"]
        out += [f"| {x['sim']:.3f} | {x['category']} | {x['a']} | {x['b']} | {x['guard']} |"
                for x in d["highest_scoring_different"]]
        out += ["", "### Near-misses the guard let through (sim ≥ 0.90)", ""]
        out += [f"- {x['sim']:.3f} [{x['category']}] {x['a']} → {x['b']}" for x in d["guard_misses"]] or ["- none"]
        out += ["", "### Paraphrases the guard wrongly rejected", ""]
        out += [f"- {x['sim']:.3f} [{x['category']}] {x['a']} → {x['b']} ({x['guard']})"
                for x in d["guard_false_rejections"]] or ["- none"]
        out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    main()
