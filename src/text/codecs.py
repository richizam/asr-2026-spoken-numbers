"""Number target representations used by training and inference.

Two modes are supported:

- ``words``: the original Russian ``num2words`` surface form.
- ``word_tokens``: the same normalized Russian words, but emitted as a
  sequence of whole-word CTC tokens rather than characters.
- ``triplet_3x3``: a fixed-width ``XYZ|ABC`` digit string where the left
  triplet is the thousands group and the right triplet is the units group.
"""

from __future__ import annotations

from dataclasses import dataclass

from .denormalizer import TrieSnapper
from .normalizer import int_to_words
from .vocab import CyrillicVocab, TripletDigitVocab, WordTokenVocab


def int_to_triplet_text(n: int) -> str:
    if not 0 <= n <= 999_999:
        raise ValueError(f"triplet target expects n in [0, 999999], got {n}")
    return f"{n // 1000:03d}|{n % 1000:03d}"


class TripletSnapper:
    """Best-effort parser for fixed-width XYZ|ABC targets."""

    def __init__(self, lo: int = 0, hi: int = 999_999) -> None:
        self.lo = lo
        self.hi = hi

    def snap(self, text: str) -> int:
        cleaned = "".join(ch for ch in text if ch.isdigit() or ch == "|")
        if not cleaned:
            return self.lo
        if "|" in cleaned:
            left_raw, right_raw = cleaned.split("|", 1)
            left = "".join(ch for ch in left_raw if ch.isdigit())[-3:].rjust(3, "0")
            right = "".join(ch for ch in right_raw if ch.isdigit())[-3:].rjust(3, "0")
            digits = left + right
        else:
            digits = "".join(ch for ch in cleaned if ch.isdigit())[-6:].rjust(6, "0")
        value = int(digits[:3]) * 1000 + int(digits[3:])
        if value < self.lo:
            return self.lo
        if value > self.hi:
            return self.hi
        return value

    def decode_to_int(self, text: str) -> int:
        return self.snap(text)


@dataclass
class WordsNumberCodec:
    vocab: CyrillicVocab
    snapper: TrieSnapper
    mode: str = "words"

    def target_text(self, n: int) -> str:
        return int_to_words(n)

    def encode_int(self, n: int) -> list[int]:
        return self.vocab.encode(self.target_text(n))

    def decode_to_int(self, text: str) -> int:
        return self.snapper.decode_to_int(text)


@dataclass
class WordTokenNumberCodec:
    vocab: WordTokenVocab
    snapper: TrieSnapper
    mode: str = "word_tokens"

    def target_text(self, n: int) -> str:
        return int_to_words(n)

    def encode_int(self, n: int) -> list[int]:
        return self.vocab.encode(self.target_text(n))

    def decode_to_int(self, text: str) -> int:
        return self.snapper.decode_to_int(text)


@dataclass
class TripletNumberCodec:
    vocab: TripletDigitVocab
    snapper: TripletSnapper
    mode: str = "triplet_3x3"

    def target_text(self, n: int) -> str:
        return int_to_triplet_text(n)

    def encode_int(self, n: int) -> list[int]:
        return self.vocab.encode(self.target_text(n))

    def decode_to_int(self, text: str) -> int:
        return self.snapper.decode_to_int(text)


def build_number_codec(mode: str = "words"):
    if mode in ("words", "num2words", "russian_words"):
        return WordsNumberCodec(vocab=CyrillicVocab(), snapper=TrieSnapper())
    if mode in ("word_tokens", "token_words", "normalized_word_tokens"):
        return WordTokenNumberCodec(vocab=WordTokenVocab(), snapper=TrieSnapper())
    if mode in ("triplet", "triplet_3x3", "xyz_abc"):
        return TripletNumberCodec(vocab=TripletDigitVocab(), snapper=TripletSnapper())
    raise ValueError(f"unknown target mode: {mode}")
