import csv
import json

import pytest

import word_stats as w


def test_normalize_unifies_spelling_variants():
    # Arabic yeh/kaf/heh -> Urdu forms; vowel marks and tatweel removed; ZWNJ removed
    assert w.normalize("كتاب") == w.normalize("کتاب")
    assert w.normalize("علمي") == w.normalize("علمی")
    assert w.normalize("سَائِنس") == "سائنس"
    assert w.normalize("اـردو") == "اردو"
    assert w.normalize("اس‌لیے") == "اسلیے"
    assert w.normalize("Newton") == "newton"


def test_normalize_keeps_distinct_letters():
    assert w.normalize("آب") == "آب"            # alef-madda stays one letter
    assert w.normalize("بھی") != w.normalize("بہی")   # do-chashmi he is not plain he
    assert w.normalize("آب") != w.normalize("اب")


def test_tokenize_drops_punctuation_and_numbers():
    toks = w.tokenize("سائنس، ٹیکنالوجی۔ سن 1969 اور ۱۹۶۹ میں؟")
    assert [t.key for t in toks] == ["سائنس", "ٹیکنالوجی", "سن", "اور", "میں"]
    toks = w.tokenize("سن 1969", keep_numbers=True)
    assert [t.key for t in toks] == ["سن", "1969"]


def test_tokenize_latin_and_mixed():
    toks = w.tokenize("آئن سٹائن Einstein's theory")
    assert [t.key for t in toks] == ["آئن", "سٹائن", "einstein's", "theory"]
    assert w.script_of("theory") == "latin" and w.script_of("سائنس") == "urdu"


def test_token_spans_point_at_original_text():
    text = "سَائِنس اور علم"
    t = w.tokenize(text)[0]
    assert text[t.start:t.end] == "سَائِنس" and t.key == "سائنس"


BOOKS = {
    "bookA": [["سائنس اور علم", "علم علم"], ["سائنس"]],
    "bookB": [["علم"]],
}


def test_counts_pages_and_forms():
    an = w.analyze(BOOKS)
    assert an.counts["bookA"]["علم"] == 3 and an.counts["bookB"]["علم"] == 1
    assert an.pages["سائنس"] == {("bookA", 1), ("bookA", 2)}
    assert an.book_pages == {"bookA": 2, "bookB": 1}


def test_concordance_rows_have_page_paragraph_and_context():
    an = w.analyze(BOOKS, width=1)
    row = next(r for r in an.rows if r[0] == "bookA" and r[1] == 1 and r[2] == 1 and r[3] == "اور")
    assert row[5] == "سائنس **اور** علم"
    # first token in a paragraph has no left context
    first = next(r for r in an.rows if r[2] == 2 and r[1] == 1)
    assert first[5].startswith("**علم**")


def test_frequency_rows_ranked_with_ties_and_per_book_columns():
    an = w.analyze(BOOKS)
    rows = w.frequency_rows(an)
    assert rows[0][:4] == [1, "علم", "urdu", 4]
    assert rows[0][4:] == [3, 1, 2]   # bookA, bookB, distinct (book,page) pairs
    sci = next(r for r in rows if r[1] == "سائنس")
    assert sci[3] == 2 and w.frequency_rows(an, min_count=3) == [rows[0]]
    tied = [r for r in rows if r[3] == 1]
    assert len({r[0] for r in tied}) == 1   # equal counts share a rank


def test_search_prints_matches(capsys):
    an = w.analyze(BOOKS)
    assert w.search(an, ["علمِ"], limit=2) == 4    # query is normalised too
    out = capsys.readouterr().out
    assert "[bookA p.1]" in out and "(showing 2" in out


def test_cli_writes_files(tmp_path):
    (tmp_path / "Book One.ocr.json").write_text(json.dumps(BOOKS["bookA"], ensure_ascii=False), encoding="utf-8")
    (tmp_path / "Book Two.ocr.json").write_text(json.dumps(BOOKS["bookB"], ensure_ascii=False), encoding="utf-8")
    w.main([str(tmp_path)])
    out = tmp_path / "word_stats"
    raw = (out / "word_frequencies.csv").read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")                      # BOM for Excel/Numbers
    rows = list(csv.reader(raw.decode("utf-8-sig").splitlines()))
    assert rows[0] == ["rank", "word", "script", "total", "Book One", "Book Two", "pages_used_on"]
    assert rows[1][1:5] == ["علم", "urdu", "4", "3"]
    conc = list(csv.reader((out / "concordance.csv").read_text(encoding="utf-8-sig").splitlines()))
    assert conc[0][:3] == ["book", "page", "paragraph"] and len(conc) == 1 + 7
    assert "distinct words" in (out / "summary.txt").read_text(encoding="utf-8")


def test_rejects_bad_json(tmp_path):
    p = tmp_path / "x.ocr.json"
    p.write_text('{"a": 1}')
    with pytest.raises(ValueError):
        w.load_pages(p)


def test_no_inputs_is_an_error(tmp_path):
    with pytest.raises(SystemExit):
        w.find_inputs([tmp_path])


def test_translate_words_caches_and_skips_latin(tmp_path):
    cache = tmp_path / "t.json"
    calls = []

    def fake(words):
        calls.append(list(words))
        return [f"EN({x}). " for x in words]

    out = w.translate_words(["علم", "einstein", "اللہ"], cache, translator=fake)
    assert out == {"علم": "en(علم)", "einstein": "einstein", "اللہ": "en(اللہ)"}
    assert calls == [["علم", "اللہ"]]               # Latin word never sent to the model
    # second run: everything cached, model not called; manual edits to the cache are honoured
    data = json.loads(cache.read_text(encoding="utf-8"))
    data["علم"] = "knowledge"
    cache.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    out2 = w.translate_words(["علم", "اللہ"], cache, translator=lambda ws: pytest.fail("cached"))
    assert out2["علم"] == "knowledge" and len(calls) == 1


def test_frequency_rows_and_summary_include_gloss():
    an = w.analyze(BOOKS)
    rows = w.frequency_rows(an, glosses={"علم": "knowledge"})
    assert rows[0][1:4] == ["علم", "knowledge", "urdu"]
    assert rows[1][2] == ""                         # untranslated words get an empty gloss
    assert "علم   = knowledge" in w.summary_text(an, glosses={"علم": "knowledge"})


def test_cli_translate_flag(tmp_path, monkeypatch):
    (tmp_path / "B.ocr.json").write_text(json.dumps(BOOKS["bookA"], ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(w, "_nllb_translator", lambda fp16: (lambda ws: [f"gloss-{x}" for x in ws]))
    w.main([str(tmp_path), "--translate", "--translate-top", "2"])
    out = tmp_path / "word_stats"
    rows = list(csv.reader((out / "word_frequencies.csv").read_text(encoding="utf-8-sig").splitlines()))
    assert rows[0][:4] == ["rank", "word", "english_gloss_machine", "script"]
    glossed = [r for r in rows[1:] if r[2]]
    assert len(glossed) == 2                        # only the top 2 words translated
    assert (out / "word_translations.json").exists()
