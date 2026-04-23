from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.decode.ctc_beam import CTCBeamDecoder
from src.text import build_number_codec


def _make_one_hot_logprobs(codec, text: str) -> torch.Tensor:
    ids = codec.vocab.encode(text)
    sequence: list[int] = []
    for i, c in enumerate(ids):
        if i > 0 and ids[i] == ids[i - 1]:
            sequence.append(codec.vocab.blank_id)
        sequence.append(c)
    lp = torch.full((len(sequence), len(codec.vocab)), -10.0)
    for t, c in enumerate(sequence):
        lp[t, c] = 0.0
    return torch.log_softmax(lp, dim=-1)


@pytest.mark.parametrize("n", [1000, 1005, 12345, 500869, 999999])
def test_word_token_roundtrip(n: int) -> None:
    codec = build_number_codec("word_tokens")
    assert codec.decode_to_int(codec.target_text(n)) == n


def test_word_token_decoder_greedy_and_trie() -> None:
    codec = build_number_codec("word_tokens")
    target = 500869
    text = codec.target_text(target)
    lp = _make_one_hot_logprobs(codec, text)
    dec = CTCBeamDecoder(vocab=codec.vocab, codec=codec, beam_width=4, trie_range=(500000, 500999))
    assert dec.greedy_decode(lp) == text
    assert dec.decode_to_int(lp, method="greedy") == target
    assert dec.decode_to_int(lp, method="beam_trie") == target
