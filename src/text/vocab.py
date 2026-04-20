"""Character vocabulary for Russian-number CTC.

Layout: index 0 is the CTC blank token. The rest are the 33 Cyrillic
letters plus a single space used as a word separator inside
num2words output.
"""

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
