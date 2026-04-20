"""Integer -> Russian words via num2words.

num2words(..., lang='ru') returns the feminine form for 1 and 2 when used
with the implicit feminine noun "тысяча" (e.g. 1000 -> "одна тысяча",
2000 -> "две тысячи"). We strip any hyphens and lowercase the result to
match our flat character vocabulary.
"""

from num2words import num2words


def int_to_words(n: int) -> str:
    text = num2words(n, lang="ru")
    return text.lower().replace("-", " ").replace("  ", " ").strip()
