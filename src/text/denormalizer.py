"""Russian number words -> integer, with word-level snap repair.

Two-stage pipeline:

1. ``words_to_int(text)`` walks the tokens with a standard group-then-thousand
   accumulator: units/teens/tens/hundreds add into a group, ``тысяч(а|и|.)``
   flushes the group as the thousands component.

2. ``TrieSnapper`` repairs a noisy CTC decode: each out-of-vocab token is
   snapped to the closest legal number word by Levenshtein distance, then
   re-parsed. Combined with a bounds clamp this guarantees the final
   answer lies in [1_000, 999_999].
"""

from __future__ import annotations

from typing import Iterable

UNITS_MASC = {
    "ноль": 0,
    "один": 1, "одна": 1, "одно": 1,
    "два": 2, "две": 2,
    "три": 3,
    "четыре": 4,
    "пять": 5,
    "шесть": 6,
    "семь": 7,
    "восемь": 8,
    "девять": 9,
}

TEENS = {
    "десять": 10,
    "одиннадцать": 11,
    "двенадцать": 12,
    "тринадцать": 13,
    "четырнадцать": 14,
    "пятнадцать": 15,
    "шестнадцать": 16,
    "семнадцать": 17,
    "восемнадцать": 18,
    "девятнадцать": 19,
}

TENS = {
    "двадцать": 20,
    "тридцать": 30,
    "сорок": 40,
    "пятьдесят": 50,
    "шестьдесят": 60,
    "семьдесят": 70,
    "восемьдесят": 80,
    "девяносто": 90,
}

HUNDREDS = {
    "сто": 100,
    "двести": 200,
    "триста": 300,
    "четыреста": 400,
    "пятьсот": 500,
    "шестьсот": 600,
    "семьсот": 700,
    "восемьсот": 800,
    "девятьсот": 900,
}

THOUSAND_MARKERS = {"тысяча", "тысячи", "тысяч"}

NUM_WORDS: dict[str, int] = {**UNITS_MASC, **TEENS, **TENS, **HUNDREDS}
ALL_NUMBER_TOKENS: list[str] = list(NUM_WORDS) + list(THOUSAND_MARKERS)


def words_to_int(text: str) -> int:
    tokens = text.lower().split()
    return _parse_tokens(tokens)


def _parse_tokens(tokens: Iterable[str]) -> int:
    current = 0
    total = 0
    for t in tokens:
        if t in THOUSAND_MARKERS:
            if current == 0:
                current = 1
            total += current * 1000
            current = 0
        elif t in NUM_WORDS:
            current += NUM_WORDS[t]
    return total + current


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]


class TrieSnapper:
    """Repairs noisy CTC transcripts into a legal integer in [lo, hi]."""

    def __init__(self, lo: int = 1_000, hi: int = 999_999) -> None:
        self.lo = lo
        self.hi = hi
        self._vocab = ALL_NUMBER_TOKENS

    def snap_token(self, tok: str) -> str | None:
        if not tok:
            return None
        if tok in NUM_WORDS or tok in THOUSAND_MARKERS:
            return tok
        best_tok, best_d = None, 10**9
        for cand in self._vocab:
            d = _levenshtein(tok, cand)
            if d < best_d:
                best_d = d
                best_tok = cand
        if best_tok is None:
            return None
        max_d = max(2, len(best_tok) // 3)
        return best_tok if best_d <= max_d else None

    def snap(self, text: str) -> int:
        tokens = text.lower().split()
        clean: list[str] = []
        for t in tokens:
            s = self.snap_token(t)
            if s is not None:
                clean.append(s)
        n = _parse_tokens(clean)
        if n < self.lo:
            n = self.lo
        elif n > self.hi:
            n = self.hi
        return n

    def decode_to_int(self, text: str) -> int:
        """Preferred API: try exact parse, fall back to word-snap."""
        tokens = text.lower().split()
        all_known = all(t in NUM_WORDS or t in THOUSAND_MARKERS for t in tokens)
        if all_known:
            n = _parse_tokens(tokens)
            if self.lo <= n <= self.hi:
                return n
        return self.snap(text)
