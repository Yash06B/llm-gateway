"""A lexical check that runs after a semantic cache match and before the cached answer is served.

Embeddings are bad at the differences that matter most for caching. "Convert Celsius to Fahrenheit" and
"Convert Fahrenheit to Celsius" score 0.997 with bge-small, higher than most real paraphrases, so no similarity
threshold separates them (see eval/results/cache.md). The guard rejects a match when the two prompts

  1. contain different numbers ("5 feet" vs "6 feet", "Python 2" vs "Python 3"),
  2. are a direction swap: the same words move across a "to"/"into" pivot ("London to New York" vs
     "New York to London", "list to set" vs "set to list"), or
  3. are the same sentence with one or two words substituted in place ("encrypt a file with GPG" vs "decrypt a
     file with GPG", "capital of Australia" vs "capital of Austria"). Spelling variants ("pyhton") don't count.

It has no word lists of antonyms or entities, so it isn't tuned to particular prompts. The cost is recall: a
paraphrase that swaps exactly one word for a synonym in place ("Who wrote X" vs "Who is the author of X") is
rejected too. The guard trades those savings for not serving wrong answers.
"""

from __future__ import annotations

import re

STOPWORDS = frozenset("""
a an the and or but if then so of in on at to from by for with without into onto about as is are was were be been
being am do does did doing have has had having can could would should will shall may might must i me my mine we us
our you your yours he him his she her it its they them their this that these those there here what whats which who
whom whose why how when where whats hows please thanks thank hi hello hey just quick question kindly briefly simply
simple terms really actually tell explain give show share help know want need like way some any much many very also
sorry basic new wondering curious been so ok okay could would get got let lets make im ive id one
""".split())

NUMBER_WORDS = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
                "seven": "7", "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12",
                "twenty": "20", "hundred": "100", "thousand": "1000", "million": "1000000",
                "first": "1", "second": "2", "third": "3", "half": "0.5", "double": "2x", "single": "1"}

_TOKEN = re.compile(r"[a-z0-9]+(?:[.+#][a-z0-9]+)*[+#]*")


def _norm(word: str) -> str:
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]
    return word


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower().replace("'s ", " ").replace("’s ", " "))


def numbers(text: str) -> list[str]:
    out = []
    for t in _tokens(text):
        if re.fullmatch(r"\d+(?:\.\d+)?", t):
            out.append(t)
        elif t in NUMBER_WORDS:
            out.append(NUMBER_WORDS[t])
    return out


def _is_content(t: str) -> bool:
    return t not in STOPWORDS and t not in NUMBER_WORDS and not t.isdigit() and len(t) > 1


def content_words(text: str) -> list[str]:
    return [_norm(t) for t in _tokens(text) if _is_content(t)]


PIVOTS = frozenset({"to", "into"})


def pivot_sides(text: str) -> dict[str, int] | None:
    """Content word -> 0 if before the first "to"/"into", 1 if after. None if there is no pivot."""
    toks = _tokens(text)
    at = next((i for i, t in enumerate(toks) if t in PIVOTS), None)
    if at is None:
        return None
    return {_norm(t): int(i > at) for i, t in enumerate(toks) if _is_content(t)}


def _edit_distance(a: str, b: str, cap: int = 3) -> int:
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def is_typo(a: str, b: str) -> bool:
    """Spelling variant rather than a different word: same first letter, similar length, edit distance <= 2.

    The first-letter and length rules are what keep encrypt/decrypt, ascending/descending and
    advantages/disadvantages from being treated as typos of each other.
    """
    if a == b:
        return True
    if a[0] != b[0] or abs(len(a) - len(b)) > 1 or min(len(a), len(b)) < 4:
        return False
    return _edit_distance(a, b) <= 2


def check(a: str, b: str) -> tuple[bool, str]:
    """Is it safe to answer prompt `b` with the cached answer to prompt `a`? Returns (ok, reason)."""
    if numbers(a) != numbers(b):
        return False, "numbers differ"
    sa, sb = pivot_sides(a), pivot_sides(b)
    if sa and sb:
        moved = [w for w in sa if w in sb and sa[w] != sb[w]]
        if moved:
            return False, f"direction swap ({', '.join(moved)} moved across 'to')"
    ca, cb = content_words(a), content_words(b)
    if len(ca) == len(cb):
        diff = [(x, y) for x, y in zip(ca, cb) if not is_typo(x, y)]
        reordered = sorted(x for x, _ in diff) == sorted(y for _, y in diff)  # "list in Python" / "Python list"
        if 0 < len(diff) <= 2 and len(diff) < len(ca) and not reordered:
            return False, "substituted " + ", ".join(f"{x} -> {y}" for x, y in diff)
    return True, "ok"
