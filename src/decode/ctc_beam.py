"""CTC decoding: greedy, prefix-beam, KenLM shallow fusion, trie snap.

Structure mirrors ``assignments/assignment2/wav2vec2decoder.py`` but is
self-contained around our character vocab and the Russian-words parser.

Beam search implementation is prefix beam search (Graves' algorithm) with
separate probability mass for hypotheses ending in blank vs. a real
character, which is the standard approach for char-level CTC.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Optional

import torch

from ..text import CyrillicVocab
from ..text.denormalizer import TrieSnapper


def _log_add(a: float, b: float) -> float:
    if a == float("-inf"):
        return b
    if b == float("-inf"):
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


@dataclass
class BeamHypothesis:
    ids: tuple[int, ...]
    log_prob: float


class CTCBeamDecoder:
    def __init__(
        self,
        vocab: Optional[CyrillicVocab] = None,
        beam_width: int = 16,
        alpha: float = 0.4,
        beta: float = 1.0,
        temperature: float = 1.0,
        lm_model_path: Optional[str] = None,
        snap: Optional[TrieSnapper] = None,
    ) -> None:
        self.vocab = vocab or CyrillicVocab()
        self.blank_id = self.vocab.blank_id
        self.space_id = self.vocab.space_id
        self.beam_width = beam_width
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.snap = snap or TrieSnapper()
        self.lm = None
        if lm_model_path is not None:
            try:
                import kenlm
                self.lm = kenlm.Model(lm_model_path)
            except ImportError:
                self.lm = None

    # ------------------------------------------------------------------
    # Core decoding methods
    # ------------------------------------------------------------------

    def greedy_decode(self, log_probs: torch.Tensor) -> str:
        """log_probs: (T, V)."""
        ids = log_probs.argmax(dim=-1).tolist()
        return self.vocab.decode_ctc(ids)

    def _beam_step(
        self,
        log_probs: torch.Tensor,
        lm_weight: float = 0.0,
    ) -> list[BeamHypothesis]:
        """Prefix beam search. Returns beams sorted by score desc."""
        T, V = log_probs.shape
        # Pb[prefix] = log prob prefix ends in blank, Pnb[prefix] = log prob ends in non-blank
        beams: dict[tuple[int, ...], tuple[float, float]] = {(): (0.0, float("-inf"))}

        for t in range(T):
            lp = log_probs[t]
            next_beams: dict[tuple[int, ...], tuple[float, float]] = defaultdict(
                lambda: (float("-inf"), float("-inf"))
            )
            for prefix, (pb, pnb) in beams.items():
                # Extend with blank
                blank_lp = float(lp[self.blank_id].item())
                nb_pb, nb_pnb = next_beams[prefix]
                nb_pb = _log_add(nb_pb, _log_add(pb, pnb) + blank_lp)
                next_beams[prefix] = (nb_pb, nb_pnb)

                # Extend with each non-blank token
                for c in range(V):
                    if c == self.blank_id:
                        continue
                    c_lp = float(lp[c].item())
                    if c_lp < -30:
                        continue  # cheap prune: skip near-zero-prob tokens
                    if prefix and prefix[-1] == c:
                        new_prefix = prefix
                        np_pb, np_pnb = next_beams[new_prefix]
                        np_pnb = _log_add(np_pnb, pnb + c_lp)
                        next_beams[new_prefix] = (np_pb, np_pnb)

                        new_prefix = prefix + (c,)
                        np_pb, np_pnb = next_beams[new_prefix]
                        np_pnb = _log_add(np_pnb, pb + c_lp)
                        next_beams[new_prefix] = (np_pb, np_pnb)
                    else:
                        new_prefix = prefix + (c,)
                        np_pb, np_pnb = next_beams[new_prefix]
                        np_pnb = _log_add(np_pnb, _log_add(pb, pnb) + c_lp)
                        next_beams[new_prefix] = (np_pb, np_pnb)

            # Prune
            scored: list[tuple[tuple[int, ...], float, float]] = []
            for prefix, (pb, pnb) in next_beams.items():
                total = _log_add(pb, pnb)
                score = total
                if lm_weight > 0.0 and self.lm is not None and len(prefix) > 0:
                    text = self.vocab.decode(list(prefix))
                    score = total + lm_weight * self._lm_score(text) + self.beta * _word_count(text)
                scored.append((prefix, pb, pnb, score))  # type: ignore[arg-type]
            scored.sort(key=lambda x: x[-1], reverse=True)
            beams = {p: (pb, pnb) for (p, pb, pnb, _s) in scored[: self.beam_width]}

        # Final ranking
        results: list[BeamHypothesis] = []
        for prefix, (pb, pnb) in beams.items():
            total = _log_add(pb, pnb)
            if lm_weight > 0.0 and self.lm is not None and len(prefix) > 0:
                text = self.vocab.decode(list(prefix))
                total = total + lm_weight * self._lm_score(text) + self.beta * _word_count(text)
            results.append(BeamHypothesis(ids=prefix, log_prob=total))
        results.sort(key=lambda h: h.log_prob, reverse=True)
        return results

    def beam_search_decode(self, log_probs: torch.Tensor, return_beams: bool = False):
        beams = self._beam_step(log_probs, lm_weight=0.0)
        if return_beams:
            return beams
        return self.vocab.decode(list(beams[0].ids)) if beams else ""

    def beam_search_with_lm(self, log_probs: torch.Tensor) -> str:
        if self.lm is None:
            return self.beam_search_decode(log_probs)
        beams = self._beam_step(log_probs, lm_weight=self.alpha)
        return self.vocab.decode(list(beams[0].ids)) if beams else ""

    def lm_rescore(self, beams: Iterable[BeamHypothesis]) -> str:
        if self.lm is None:
            return self.vocab.decode(list(next(iter(beams)).ids))
        best = None
        best_score = float("-inf")
        for h in beams:
            text = self.vocab.decode(list(h.ids))
            score = h.log_prob + self.alpha * self._lm_score(text) + self.beta * _word_count(text)
            if score > best_score:
                best_score = score
                best = text
        return best or ""

    def _lm_score(self, text: str) -> float:
        if self.lm is None or not text.strip():
            return 0.0
        return float(self.lm.score(text, bos=True, eos=True))

    # ------------------------------------------------------------------
    # High-level wrapper: log-probs -> final integer
    # ------------------------------------------------------------------

    def decode_text(self, log_probs: torch.Tensor, method: str = "beam_lm") -> str:
        log_probs = log_probs / self.temperature
        if method == "greedy":
            text = self.greedy_decode(log_probs)
        elif method == "beam":
            text = self.beam_search_decode(log_probs)  # type: ignore[assignment]
        elif method == "beam_lm":
            text = self.beam_search_with_lm(log_probs)
        elif method == "beam_lm_rescore":
            beams = self.beam_search_decode(log_probs, return_beams=True)
            text = self.lm_rescore(beams)  # type: ignore[arg-type]
        else:
            raise ValueError(f"unknown method: {method}")
        return text

    def decode_to_int(self, log_probs: torch.Tensor, method: str = "beam_lm") -> int:
        text = self.decode_text(log_probs, method=method)
        return self.snap.decode_to_int(text)


def _word_count(text: str) -> int:
    return len(text.split())
