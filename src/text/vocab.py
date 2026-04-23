"""Vocabularies for Russian-number CTC."""

from .denormalizer import NUM_WORDS, THOUSAND_MARKERS

RUSSIAN_ALPHABET = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
SPACE = " "
BLANK = "<blank>"


class CyrillicVocab:
    def __init__(self) -> None:
        self.tokens = [BLANK, SPACE] + list(RUSSIAN_ALPHABET)
        self.tok2id = {t: i for i, t in enumerate(self.tokens)}
        self.id2tok = {i: t for t, i in self.tok2id.items()}
        self.blank_id = self.tok2id[BLANK]
        self.space_id = self.tok2id[SPACE]

    def __len__(self) -> int:
        return len(self.tokens)

    def encode(self, text: str) -> list[int]:
        return [self.tok2id[c] for c in text if c in self.tok2id]

    def decode(self, ids: list[int]) -> str:
        out = []
        for i in ids:
            t = self.id2tok[i]
            if t == BLANK:
                continue
            out.append(t)
        return "".join(out)

    def decode_ctc(self, ids: list[int]) -> str:
        """Collapse repeats then drop blanks (standard CTC collapse)."""
        collapsed: list[int] = []
        prev = -1
        for i in ids:
            if i != prev:
                collapsed.append(i)
            prev = i
        return self.decode([i for i in collapsed if i != self.blank_id])


DIGITS = "0123456789"
SEPARATOR = "|"
WORD_TOKENS = tuple(dict.fromkeys([*NUM_WORDS.keys(), *sorted(THOUSAND_MARKERS)]))


class TripletDigitVocab:
    """CTC vocabulary for fixed-width XYZ|ABC targets."""

    def __init__(self) -> None:
        self.tokens = [BLANK, SEPARATOR] + list(DIGITS)
        self.tok2id = {t: i for i, t in enumerate(self.tokens)}
        self.id2tok = {i: t for t, i in self.tok2id.items()}
        self.blank_id = self.tok2id[BLANK]
        self.separator_id = self.tok2id[SEPARATOR]

    def __len__(self) -> int:
        return len(self.tokens)

    def encode(self, text: str) -> list[int]:
        return [self.tok2id[c] for c in text if c in self.tok2id]

    def decode(self, ids: list[int]) -> str:
        out = []
        for i in ids:
            t = self.id2tok[i]
            if t == BLANK:
                continue
            out.append(t)
        return "".join(out)

    def decode_ctc(self, ids: list[int]) -> str:
        collapsed: list[int] = []
        prev = -1
        for i in ids:
            if i != prev:
                collapsed.append(i)
            prev = i
        return self.decode([i for i in collapsed if i != self.blank_id])


class WordTokenVocab:
    """CTC vocabulary over whole Russian number words."""

    def __init__(self) -> None:
        self.tokens = [BLANK] + list(WORD_TOKENS)
        self.tok2id = {t: i for i, t in enumerate(self.tokens)}
        self.id2tok = {i: t for t, i in self.tok2id.items()}
        self.blank_id = self.tok2id[BLANK]
        self.word_level = True

    def __len__(self) -> int:
        return len(self.tokens)

    def encode(self, text: str) -> list[int]:
        return [self.tok2id[t] for t in text.split() if t in self.tok2id]

    def decode(self, ids: list[int]) -> str:
        out = []
        for i in ids:
            t = self.id2tok[i]
            if t == BLANK:
                continue
            out.append(t)
        return " ".join(out)

    def decode_ctc(self, ids: list[int]) -> str:
        collapsed: list[int] = []
        prev = -1
        for i in ids:
            if i != prev:
                collapsed.append(i)
            prev = i
        return self.decode([i for i in collapsed if i != self.blank_id])
