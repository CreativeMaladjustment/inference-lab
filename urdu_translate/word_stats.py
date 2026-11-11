#!/usr/bin/env python3
"""Word frequencies + concordance from the Urdu OCR caches (*.ocr.json) written by
translate_urdu_pdf.py --save-ocr.  Pure standard library; runs in seconds.

Outputs (in --output-dir, default <input folder>/word_stats/):
  word_frequencies.csv   rank, word, script, total, one count column per book, pages
  concordance.csv        every occurrence: book, page, paragraph, word, form, context
  summary.txt            totals and the most frequent words

Page numbers are PDF page numbers (1 = first page of the PDF), the same numbering as the
"## Page N of ..." headings in the translated .md files.

  python word_stats.py ~/temp/pdfs-to-process
  python word_stats.py ~/temp/pdfs-to-process --search سائنس        # look a word up
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

# Marks that vary between sources and shouldn't make two spellings count as different words:
# Arabic/Urdu vowel marks and Quranic annotation signs, tatweel, and invisible formatting
# characters (incl. zero-width non-joiner, which is removed, not replaced by a space).
_STRIP = re.compile(
    "[ؐ-ًؚ-ٰٟۖ-ۭـ​-‏‪-‮⁦-⁩﻿]")
# Arabic-script letter variants -> the Urdu form. (ھ do-chashmi he U+06BE is a distinct Urdu
# letter and is deliberately NOT merged.)
_FOLD = str.maketrans({"ي": "ی", "ى": "ی", "ك": "ک", "ه": "ہ"})
_MARKS = "ؐ-ًؚ-ٰٟۖ-ۭـ‌‍"
_LATIN_WORD = r"[A-Za-z]+(?:['’][A-Za-z]+)*"
_WORD = re.compile(rf"{_LATIN_WORD}|(?:[^\W\d_]|[{_MARKS}])+")
_WORD_NUM = re.compile(rf"{_LATIN_WORD}|(?:[^\W_]|[{_MARKS}])+")  # --keep-numbers
_ARABIC = re.compile("[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")


def normalize(token: str) -> str:
    """Canonical form used for counting (NFKC, no vowel marks, unified yeh/kaf/heh, casefolded)."""
    t = unicodedata.normalize("NFKC", token)
    t = _STRIP.sub("", t).translate(_FOLD)
    return t.casefold()


def script_of(word: str) -> str:
    return "urdu" if _ARABIC.search(word) else ("latin" if word.isascii() else "other")


@dataclass
class Token:
    start: int
    end: int
    key: str


def tokenize(text: str, keep_numbers: bool = False) -> list[Token]:
    rx = _WORD_NUM if keep_numbers else _WORD
    out = []
    for m in rx.finditer(text):
        key = normalize(m.group())
        if key:
            out.append(Token(m.start(), m.end(), key))
    return out


def kwic(text: str, toks: list[Token], i: int, width: int) -> str:
    """Context: `width` tokens either side of token i, target wrapped in **...**."""
    a, b = toks[max(0, i - width)].start, toks[min(len(toks) - 1, i + width)].end
    t = toks[i]
    return f"{text[a:t.start]}**{text[t.start:t.end]}**{text[t.end:b]}".strip()


# ------------------------------------------------------------------ loading / analysis
def load_pages(path: Path) -> list[list[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not (isinstance(data, list) and all(isinstance(p, list) for p in data)):
        raise ValueError(f"{path.name}: expected a list of pages, each a list of paragraphs")
    return data


def book_name(path: Path) -> str:
    name = path.name
    return name[: -len(".ocr.json")] if name.endswith(".ocr.json") else path.stem


def find_inputs(paths: list[Path]) -> list[Path]:
    found: list[Path] = []
    for p in paths:
        p = p.expanduser()
        if p.is_dir():
            found += sorted(p.glob("*.ocr.json"))
        elif p.is_file():
            found.append(p)
        else:
            sys.exit(f"Not found: {p}")
    if not found:
        sys.exit("No *.ocr.json files found (run translate_urdu_pdf.py with --save-ocr first).")
    return found


@dataclass
class Analysis:
    books: list[str]
    counts: dict[str, Counter]              # book -> Counter(word)
    pages: dict[str, set]                   # word -> {(book, page)}
    forms: dict[str, Counter]               # word -> Counter(surface form)  (not written, for tests)
    book_pages: dict[str, int]
    rows: list[tuple]                       # concordance rows


def analyze(books: dict[str, list[list[str]]], width: int = 8, keep_numbers: bool = False,
            want_rows: bool = True) -> Analysis:
    counts: dict[str, Counter] = {}
    pages: dict[str, set] = defaultdict(set)
    forms: dict[str, Counter] = defaultdict(Counter)
    rows: list[tuple] = []
    for book, book_pages in books.items():
        c = counts[book] = Counter()
        for pno, paras in enumerate(book_pages, start=1):
            for qno, para in enumerate(paras, start=1):
                toks = tokenize(para, keep_numbers)
                for i, t in enumerate(toks):
                    c[t.key] += 1
                    pages[t.key].add((book, pno))
                    forms[t.key][para[t.start:t.end]] += 1
                    if want_rows:
                        rows.append((book, pno, qno, t.key, para[t.start:t.end], kwic(para, toks, i, width)))
    return Analysis(list(books), counts, pages, forms, {b: len(p) for b, p in books.items()}, rows)


# ------------------------------------------------------------------ output
def write_csv(path: Path, header: list[str], rows) -> None:
    # utf-8-sig: the BOM makes Excel/Numbers read Urdu correctly
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def frequency_rows(an: Analysis, min_count: int = 1, glosses: dict | None = None) -> list[list]:
    total = Counter()
    for c in an.counts.values():
        total.update(c)
    ordered = sorted(total.items(), key=lambda kv: (-kv[1], kv[0]))
    rows, rank, prev, shown = [], 0, None, 0
    for word, n in ordered:
        shown += 1
        if n != prev:
            rank, prev = shown, n   # equal counts share a rank ("competition" ranking)
        if n >= min_count:
            row = ([rank, word, script_of(word), n]
                   + [an.counts[b][word] for b in an.books] + [len(an.pages[word])])
            if glosses is not None:                    # English sits next to the word
                row.insert(2, glosses.get(word, ""))
            rows.append(row)
    return rows


# ------------------------------------------------------------------ English glosses
def translate_words(words: list[str], cache_path: Path, translator=None, fp16: bool = False) -> dict:
    """Machine-translate single words (Urdu -> English) with NLLB. Results are cached in
    `cache_path` (JSON) so re-runs are instant; edit that file to correct a gloss.
    Latin-script words are copied as they are. `translator(list[str]) -> list[str]` is
    injectable for tests; the default loads the NLLB model."""
    cache: dict = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    for wd in words:
        if script_of(wd) != "urdu":
            cache.setdefault(wd, wd)
    todo = [wd for wd in words if wd not in cache]
    if todo:
        if translator is None:
            translator = _nllb_translator(fp16)
        print(f"Translating {len(todo)} word(s) with NLLB ({len(cache)} already cached)")
        for wd, en in zip(todo, translator(todo)):
            cache[wd] = en.strip().rstrip(".").lower()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1, sort_keys=True),
                              encoding="utf-8")
    return {wd: cache[wd] for wd in words}


def _nllb_translator(fp16: bool):
    import translate_urdu_pdf as tu   # heavy import (torch); only when --translate is used

    tok, model = tu.load_translator(fp16)

    def run(words: list[str]) -> list[str]:
        chunks = [tu.Chunk(0, i, wd) for i, wd in enumerate(words)]
        # single words: short inputs; NLLB can still ramble on a non-word, so treat glosses as hints
        return tu.translate_chunks(chunks, tok, model, batch_size=16, max_input_tokens=16, num_beams=4)

    return run


def summary_text(an: Analysis, top: int = 25, glosses: dict | None = None) -> str:
    lines = []
    total = Counter()
    for b in an.books:
        c = an.counts[b]
        total.update(c)
        hapax = sum(1 for n in c.values() if n == 1)
        lines.append(f"{b}: {an.book_pages[b]} pages, {sum(c.values())} word tokens, "
                     f"{len(c)} distinct words, {hapax} used once")
    lines.append(f"ALL: {sum(total.values())} tokens, {len(total)} distinct words")
    lines.append("")
    lines.append(f"Top {top} words overall:")
    for word, n in sorted(total.items(), key=lambda kv: (-kv[1], kv[0]))[:top]:
        gloss = f"   = {glosses[word]}" if glosses and glosses.get(word) else ""
        lines.append(f"  {n:6d}  {word}{gloss}")
    return "\n".join(lines) + "\n"


def search(an: Analysis, queries: list[str], limit: int) -> int:
    wanted = {normalize(q) for q in queries}
    hits = [r for r in an.rows if r[3] in wanted]
    for book, page, para, word, form, ctx in hits[:limit]:
        print(f"[{book} p.{page}] {ctx}")
    more = len(hits) - limit
    print(f"\n{len(hits)} occurrence(s) of {', '.join(sorted(wanted))}"
          + (f" (showing {limit}; use --limit to see more)" if more > 0 else ""))
    return len(hits)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Word frequencies and concordance from *.ocr.json files.")
    ap.add_argument("inputs", nargs="+", type=Path, help="Folder(s) containing *.ocr.json, or the files")
    ap.add_argument("--output-dir", type=Path, help="Default: <first input folder>/word_stats")
    ap.add_argument("--context", type=int, default=8, help="Words of context either side (default 8)")
    ap.add_argument("--min-count", type=int, default=1, help="Hide words used fewer times in the frequency CSV")
    ap.add_argument("--keep-numbers", action="store_true", help="Count numerals as words")
    ap.add_argument("--translate", action="store_true",
                    help="Add an English gloss (NLLB, single-word, machine-made) for the most frequent words")
    ap.add_argument("--translate-top", type=int, default=300,
                    help="How many of the most frequent words to translate (default 300)")
    ap.add_argument("--fp16", action="store_true", help="Half-precision NLLB (less RAM)")
    ap.add_argument("--search", nargs="+", metavar="WORD", help="Print where these words occur and exit")
    ap.add_argument("--limit", type=int, default=50, help="Max --search results printed")
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    files = find_inputs(a.inputs)
    books = {}
    for f in files:
        name = book_name(f)
        if name in books:
            sys.exit(f"Two inputs share the name {name!r}")
        books[name] = load_pages(f)
    an = analyze(books, a.context, a.keep_numbers, want_rows=True)

    if a.search:
        search(an, a.search, a.limit)
        return

    first = a.inputs[0].expanduser()
    out_dir = a.output_dir or ((first if first.is_dir() else first.parent) / "word_stats")
    out_dir.mkdir(parents=True, exist_ok=True)
    glosses = None
    if a.translate:
        total = Counter()
        for c in an.counts.values():
            total.update(c)
        top_words = [wd for wd, _ in sorted(total.items(), key=lambda kv: (-kv[1], kv[0]))[:a.translate_top]]
        glosses = translate_words(top_words, out_dir / "word_translations.json", fp16=a.fp16)
    header = ["rank", "word", "script", "total"] + an.books + ["pages_used_on"]
    if glosses is not None:
        header.insert(2, "english_gloss_machine")
    write_csv(out_dir / "word_frequencies.csv", header, frequency_rows(an, a.min_count, glosses))
    write_csv(out_dir / "concordance.csv",
              ["book", "page", "paragraph", "word", "form_in_text", "context"], an.rows)
    text = summary_text(an, glosses=glosses)
    (out_dir / "summary.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"Wrote word_frequencies.csv, concordance.csv ({len(an.rows)} rows), summary.txt in {out_dir}")


if __name__ == "__main__":
    main()
