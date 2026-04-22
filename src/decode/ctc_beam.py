"""CTC decoding: greedy, prefix-beam, KenLM shallow fusion, trie snap.

Structure mirrors ``assignments/assignment2/wav2vec2decoder.py`` but is
self-contained around our character vocab and the Russian-words parser.

Beam search implementation is prefix beam search (Graves' algorithm) with
separate probability mass for hypotheses ending in blank vs. a real
character, which is the standard approach for char-level CTC.

The trie-constrained variants (``beam_search_decode_trie``,
``beam_search_with_lm_trie``) prune any hypothesis whose decoded prefix
is not a prefix of some valid ``num2words(n, 'ru')`` for n ∈ [1_000,
999_999]. This shrinks the effective output space from ~35^T down to
the 999k closed set of legal Russian number transcriptions, which is
the main leverage point when the acoustic model is uncertain on OOD
speakers.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Optional

import torch

from ..text import CyrillicVocab
from ..text.denormalizer import TrieSnapper


# Sentinel placed at terminal trie nodes. ``object()`` guarantees this key
# cannot collide with any Cyrillic char or space that lives alongside it
# in the node dicts.
_TRIE_END = object()


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


class NumberCharTrie:
    """Character-level prefix trie over num2words(n, 'ru') for n ∈ [lo, hi].

    Each node is a plain ``dict`` mapping a single character to its child
    node; terminal nodes additionally carry ``_TRIE_END`` so we can tell
    in O(1) whether the current path is a complete legal transcription.

    Build cost is dominated by ``num2words`` itself (~50 µs/call), so the
    default range of 999k strings takes ~1 min on first instantiation.
    Construction is lazy from the decoder's point of view — only users
    that actually call the trie-constrained methods pay that cost.
    """

    def __init__(self, lo: int = 1_000, hi: int = 999_999) -> None:
        from ..text import int_to_words
        self.lo = lo
        self.hi = hi
        self.root: dict = {}
        for n in range(lo, hi + 1):
            s = int_to_words(n)
            node = self.root
            for c in s:
                node = node.setdefault(c, {})
            node[_TRIE_END] = True

    @staticmethod
    def step(node: dict | None, char: str) -> dict | None:
        if node is None:
            return None
        return node.get(char)

    @staticmethod
    def is_terminal(node: dict | None) -> bool:
        return node is not None and _TRIE_END in node


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
        trie_range: tuple[int, int] = (1_000, 999_999),
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
        # Trie is built lazily on first call to a ``*_trie`` method so
        # existing callers of greedy / plain beam / beam_lm pay zero cost.
        self._trie_range = trie_range
        self._number_trie: NumberCharTrie | None = None

    def _ensure_trie(self) -> NumberCharTrie:
        if self._number_trie is None:
            self._number_trie = NumberCharTrie(*self._trie_range)
        return self._number_trie

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

    def _beam_step_trie(
        self,
        log_probs: torch.Tensor,
        lm_weight: float = 0.0,
    ) -> list[BeamHypothesis]:
        """Prefix beam search restricted to trie-valid prefixes.

        Mirrors ``_beam_step`` exactly in its CTC bookkeeping (Pb / Pnb
        split, blank extension, repeat-vs-new-char cases) but drops any
        extension whose resulting decoded prefix is not a prefix of some
        valid ``num2words(n, 'ru')``. Blank extensions and repeat
        extensions (collapsed by CTC) do not move along the trie, so
        they are always allowed; only genuine new characters must pass
        the trie step.

        Final ranking prefers terminal hypotheses (full legal
        transcriptions) over non-terminal ones of the same score —
        otherwise a very short legal decode can lose the top slot to a
        slightly-higher-scoring incomplete prefix.
        """
        trie = self._ensure_trie()
        T, V = log_probs.shape
        NEG_INF = float("-inf")
        # prefix -> (pb, pnb, trie_node)
        beams: dict[tuple[int, ...], tuple[float, float, dict]] = {
            (): (0.0, NEG_INF, trie.root)
        }

        for t in range(T):
            lp = log_probs[t]
            # Use a list [pb, pnb, node] so we can update in place without
            # rebuilding tuples on every accumulator touch.
            next_beams: dict[tuple[int, ...], list] = {}
            blank_lp = float(lp[self.blank_id].item())

            for prefix, (pb, pnb, node) in beams.items():
                # Blank extension — prefix and trie node unchanged.
                entry = next_beams.get(prefix)
                if entry is None:
                    entry = [NEG_INF, NEG_INF, node]
                    next_beams[prefix] = entry
                entry[0] = _log_add(entry[0], _log_add(pb, pnb) + blank_lp)

                for c in range(V):
                    if c == self.blank_id:
                        continue
                    c_lp = float(lp[c].item())
                    if c_lp < -30:
                        continue
                    char = self.vocab.id2tok[c]

                    if prefix and prefix[-1] == c:
                        # Repeat of last emitted char:
                        # (A) collapses into the existing prefix (no trie move).
                        entry = next_beams.get(prefix)
                        if entry is None:
                            entry = [NEG_INF, NEG_INF, node]
                            next_beams[prefix] = entry
                        entry[1] = _log_add(entry[1], pnb + c_lp)

                        # (B) a fresh occurrence, which means there was a
                        # blank just before — this *is* a new character
                        # in the decoded string, so it must take a trie
                        # step.
                        new_node = node.get(char)
                        if new_node is None:
                            continue
                        new_prefix = prefix + (c,)
                        entry = next_beams.get(new_prefix)
                        if entry is None:
                            entry = [NEG_INF, NEG_INF, new_node]
                            next_beams[new_prefix] = entry
                        entry[1] = _log_add(entry[1], pb + c_lp)
                    else:
                        new_node = node.get(char)
                        if new_node is None:
                            continue
                        new_prefix = prefix + (c,)
                        entry = next_beams.get(new_prefix)
                        if entry is None:
                            entry = [NEG_INF, NEG_INF, new_node]
                            next_beams[new_prefix] = entry
                        entry[1] = _log_add(entry[1], _log_add(pb, pnb) + c_lp)

            # Prune to beam_width using (optionally LM-augmented) total score.
            scored: list[tuple] = []
            for prefix, (pb, pnb, nd) in next_beams.items():
                total = _log_add(pb, pnb)
                score = total
                if lm_weight > 0.0 and self.lm is not None and len(prefix) > 0:
                    text = self.vocab.decode(list(prefix))
                    score = (
                        total
                        + lm_weight * self._lm_score(text)
                        + self.beta * _word_count(text)
                    )
                scored.append((prefix, pb, pnb, nd, score))
            scored.sort(key=lambda x: x[-1], reverse=True)
            beams = {p: (pb, pnb, nd) for (p, pb, pnb, nd, _s) in scored[: self.beam_width]}

        # Final ranking: split terminal vs. non-terminal so a complete
        # legal decode wins ties against an incomplete-but-slightly-higher
        # prefix. Non-terminal hypotheses still get returned as fallback
        # so the TrieSnapper can repair them downstream.
        terminal: list[BeamHypothesis] = []
        non_terminal: list[BeamHypothesis] = []
        for prefix, (pb, pnb, nd) in beams.items():
            total = _log_add(pb, pnb)
            if lm_weight > 0.0 and self.lm is not None and len(prefix) > 0:
                text = self.vocab.decode(list(prefix))
                total = (
                    total
                    + lm_weight * self._lm_score(text)
                    + self.beta * _word_count(text)
                )
            hyp = BeamHypothesis(ids=prefix, log_prob=total)
            (terminal if NumberCharTrie.is_terminal(nd) else non_terminal).append(hyp)
        terminal.sort(key=lambda h: h.log_prob, reverse=True)
        non_terminal.sort(key=lambda h: h.log_prob, reverse=True)
        return terminal + non_terminal

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

    def beam_search_decode_trie(
        self, log_probs: torch.Tensor, return_beams: bool = False
    ):
        beams = self._beam_step_trie(log_probs, lm_weight=0.0)
        if return_beams:
            return beams
        return self.vocab.decode(list(beams[0].ids)) if beams else ""

    def beam_search_with_lm_trie(self, log_probs: torch.Tensor) -> str:
        if self.lm is None:
            return self.beam_search_decode_trie(log_probs)
        beams = self._beam_step_trie(log_probs, lm_weight=self.alpha)
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
        elif method == "beam_trie":
            text = self.beam_search_decode_trie(log_probs)  # type: ignore[assignment]
        elif method == "beam_lm_trie":
            text = self.beam_search_with_lm_trie(log_probs)
        else:
            raise ValueError(f"unknown method: {method}")
        return text

    def decode_to_int(self, log_probs: torch.Tensor, method: str = "beam_lm") -> int:
        text = self.decode_text(log_probs, method=method)
        return self.snap.decode_to_int(text)


def _word_count(text: str) -> int:
    return len(text.split())
