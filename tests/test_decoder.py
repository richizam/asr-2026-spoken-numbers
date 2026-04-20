"""Decoder smoke tests with hand-crafted log-probs."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.decode.ctc_beam import CTCBeamDecoder
from src.text import CyrillicVocab, int_to_words


def _make_one_hot_logprobs(vocab: CyrillicVocab, text: str) -> torch.Tensor:
    """Build log-probs that are peaky on the target char sequence
    with blanks interleaved — greedy decode should recover ``text``."""
    ids = vocab.encode(text)
    # One blank between each char to avoid the CTC repeat-collapse issue.
    sequence: list[int] = []
    for i, c in enumerate(ids):
        if i > 0 and ids[i] == ids[i - 1]:
            sequence.append(vocab.blank_id)
        sequence.append(c)
    T = len(sequence)
    V = len(vocab)
    lp = torch.full((T, V), -10.0)
    for t, c in enumerate(sequence):
        lp[t, c] = 0.0
    return torch.log_softmax(lp, dim=-1)


def test_greedy_recovers_text() -> None:
    vocab = CyrillicVocab()
    text = int_to_words(139473)
    lp = _make_one_hot_logprobs(vocab, text)
    dec = CTCBeamDecoder(vocab=vocab)
    assert dec.greedy_decode(lp) == text


def test_decode_to_int_greedy() -> None:
    vocab = CyrillicVocab()
    text = int_to_words(500869)
    lp = _make_one_hot_logprobs(vocab, text)
    dec = CTCBeamDecoder(vocab=vocab)
    assert dec.decode_to_int(lp, method="greedy") == 500869


def test_beam_matches_greedy_on_peaky() -> None:
    vocab = CyrillicVocab()
    text = int_to_words(12345)
    lp = _make_one_hot_logprobs(vocab, text)
    dec = CTCBeamDecoder(vocab=vocab, beam_width=4)
    assert dec.beam_search_decode(lp) == text


def test_snap_recovers_with_noise() -> None:
    """If the argmax has a typo, trie snap should still produce the right int."""
    vocab = CyrillicVocab()
    ids = vocab.encode(int_to_words(100000))  # 'сто тысяч'
    # Drop the last char 'ч' -> 'сто тыся'
    ids = ids[:-1]
    sequence: list[int] = []
    for i, c in enumerate(ids):
        if i > 0 and ids[i] == ids[i - 1]:
            sequence.append(vocab.blank_id)
        sequence.append(c)
    T = len(sequence)
    V = len(vocab)
    lp = torch.full((T, V), -10.0)
    for t, c in enumerate(sequence):
        lp[t, c] = 0.0
    lp = torch.log_softmax(lp, dim=-1)
    dec = CTCBeamDecoder(vocab=vocab)
    assert dec.decode_to_int(lp, method="greedy") == 100000
