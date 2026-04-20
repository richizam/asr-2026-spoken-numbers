"""Roundtrip tests: int -> num2words -> words_to_int must be identity."""

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.text import int_to_words, words_to_int
from src.text.denormalizer import TrieSnapper


@pytest.mark.parametrize("n", [
    1000, 1001, 1005, 1021, 2000, 2005, 5000, 5005,
    10000, 21000, 100000, 100001, 139473, 500869, 999999,
])
def test_roundtrip_known(n: int) -> None:
    assert words_to_int(int_to_words(n)) == n


def test_roundtrip_all_range_sampled() -> None:
    rng = random.Random(0)
    for _ in range(10_000):
        n = rng.randint(1000, 999_999)
        assert words_to_int(int_to_words(n)) == n, f"{n} -> {int_to_words(n)}"


def test_trie_snap_exact() -> None:
    snap = TrieSnapper()
    assert snap.decode_to_int("сто тридцать девять тысяч четыреста семьдесят три") == 139473


def test_trie_snap_noisy_char_drop() -> None:
    snap = TrieSnapper()
    # drop a letter from "тысяч"
    noisy = "сто тридцать девять тысч четыреста семьдесят три"
    assert snap.decode_to_int(noisy) == 139473


def test_trie_snap_noisy_word_replace() -> None:
    snap = TrieSnapper()
    # typo in "девять" -> "девят"
    noisy = "сто тридцать девят тысяч четыреста семьдесят три"
    assert snap.decode_to_int(noisy) == 139473


def test_trie_snap_clamp_lo() -> None:
    snap = TrieSnapper()
    assert snap.decode_to_int("пять") == 1000  # below lo, clamped


def test_trie_snap_clamp_hi() -> None:
    snap = TrieSnapper()
    # Construct something that would parse above hi.
    assert snap.decode_to_int("девятьсот девяносто девять тысяч девятьсот девяносто девять") == 999999
