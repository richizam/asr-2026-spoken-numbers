"""Generate a Russian-word KenLM corpus from num2words expansions.

Writes a text file with one num2words(n, lang='ru') per line for every
integer n in [lo, hi], then runs KenLM's ``lmplz`` to train a 3-gram LM.

Usage:
    python src/lm/build_lm.py --out-corpus lm_corpus.txt --lo 1000 --hi 999999
    # then externally:
    lmplz -o 3 < lm_corpus.txt > lm.arpa
    build_binary lm.arpa lm.bin
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from ..text import int_to_words


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-corpus", required=True, type=Path)
    ap.add_argument("--lo", type=int, default=1_000)
    ap.add_argument("--hi", type=int, default=999_999)
    args = ap.parse_args()

    args.out_corpus.parent.mkdir(parents=True, exist_ok=True)
    with args.out_corpus.open("w", encoding="utf-8") as f:
        for n in tqdm(range(args.lo, args.hi + 1)):
            f.write(int_to_words(n) + "\n")
    print(f"Wrote {args.hi - args.lo + 1} lines to {args.out_corpus}")
    print("Next: lmplz -o 3 --text", args.out_corpus, "--arpa lm.arpa")


if __name__ == "__main__":
    main()
