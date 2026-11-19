"""Checks that catch OCR garbage and machine-translation hallucination.

NLLB is a translate-only model: fed noise it still emits fluent English (invented
"stories", or loops like "nom nom nom"). These pure-text checks let the pipeline say
"illegible" / "unreliable" instead of passing that output through as a translation.
"""
from __future__ import annotations

import re
from collections import Counter

ILLEGIBLE = "[illegible]"
UNRELIABLE = "[translation unreliable]"

_LOOP = re.compile(r"(.{2,12}?)\1{4,}")        # a short string repeated 5+ times in a row
_CJK = re.compile(r"[぀-ヿ㐀-鿿가-힯]")
_LETTER = re.compile(r"[^\W\d_]")


def is_degenerate(text: str, min_words: int = 6) -> bool:
    """True if the text is dominated by repetition: a word looping, one word making up
    a big share of the text, very few distinct words, or a short string repeated."""
    words = text.split()
    if _LOOP.search(text.replace(" ", "")) and len(words) >= 3:
        return True
    if len(words) < min_words:
        return False
    counts = Counter(w.lower() for w in words)
    if counts.most_common(1)[0][1] / len(words) > 0.3:
        return True
    if len(counts) / len(words) < 0.4:
        return True
    run = best = 1
    for a, b in zip(words, words[1:]):
        run = run + 1 if a.lower() == b.lower() else 1
        best = max(best, run)
    return best >= 4


def is_illegible_source(text: str) -> bool:
    """OCR text that cannot be real Urdu prose: degenerate repetition, or a large share
    of letters in scripts that never occur in these books (CJK from misread strokes)."""
    if is_degenerate(text):
        return True
    letters = _LETTER.findall(text)
    return bool(letters) and len(_CJK.findall(text)) / len(letters) > 0.2


def translation_unreliable(source: str, output: str) -> bool:
    """True if the English looks hallucinated: it loops, or its length is out of
    proportion to the Urdu it came from."""
    if is_degenerate(output, min_words=5):
        return True
    src_words, out_words = len(source.split()), len(output.split())
    if src_words >= 4 and not 0.4 <= out_words / src_words <= 2.5:
        return True
    return src_words >= 1 and out_words == 0
