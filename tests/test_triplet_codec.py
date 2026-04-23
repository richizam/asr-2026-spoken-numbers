from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.decode.ctc_beam import CTCBeamDecoder
from src.text import build_number_codec, int_to_triplet_text


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


@pytest.mark.parametrize("n", [394, 1000, 1005, 1021, 12345, 500869, 999999])
def test_triplet_roundtrip(n: int) -> None:
    codec = build_number_codec("triplet_3x3")
    assert codec.decode_to_int(int_to_triplet_text(n)) == n


def test_triplet_snap_recovers_missing_padding() -> None:
    codec = build_number_codec("triplet_3x3")
    assert codec.decode_to_int("1|5") == 1005
    assert codec.decode_to_int("12345") == 12345


def test_triplet_decoder_greedy_and_trie() -> None:
    codec = build_number_codec("triplet_3x3")
    target = 500869
    text = int_to_triplet_text(target)
    lp = _make_one_hot_logprobs(codec, text)
    dec = CTCBeamDecoder(vocab=codec.vocab, codec=codec, beam_width=4)
    assert dec.greedy_decode(lp) == text
    assert dec.decode_to_int(lp, method="greedy") == target
    assert dec.decode_to_int(lp, method="beam_trie") == target
