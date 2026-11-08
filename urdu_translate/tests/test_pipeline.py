"""Model-free tests: OCR and NLLB are replaced with stubs."""
import json
import sys
from types import SimpleNamespace

import pytest

import translate_urdu_pdf as t


def line(text, y0, y1, x0=0, x1=100):
    return SimpleNamespace(text=text, bbox=(x0, y0, x1, y1))


# ---- device -------------------------------------------------------------
def test_get_device_is_mps_or_cpu():
    dev = t.get_device()
    assert dev.type in {"mps", "cpu"}
    import os
    assert os.environ["TORCH_DEVICE"] == dev.type


def test_get_device_prefers_mps(monkeypatch):
    monkeypatch.setattr(t.torch.backends.mps, "is_available", lambda: True)
    assert t.get_device().type == "mps"


def test_get_device_cpu_fallback(monkeypatch):
    monkeypatch.setattr(t.torch.backends.mps, "is_available", lambda: False)
    assert t.get_device().type == "cpu"


# ---- paragraph grouping --------------------------------------------------
def test_lines_group_into_paragraphs_by_gap():
    lines = [
        line("پہلی لائن", 0, 20),
        line("دوسری لائن", 22, 42),
        line("نیا پیراگراف", 100, 120),  # big gap -> new paragraph
    ]
    assert t.lines_to_paragraphs(lines) == ["پہلی لائن دوسری لائن", "نیا پیراگراف"]


def test_lines_sorted_top_to_bottom_and_blank_skipped():
    lines = [line("دوسری", 22, 42), line("   ", 50, 60), line("پہلی", 0, 20)]
    assert t.lines_to_paragraphs(lines) == ["پہلی دوسری"]


def test_no_lines():
    assert t.lines_to_paragraphs([]) == []


# ---- chunking -------------------------------------------------------------
def test_chunks_respect_size_and_paragraph_boundaries():
    sentence = "یہ ایک جملہ ہے۔ "
    para = sentence * 40
    chunks = t.build_chunks([[para, "مختصر"], []], chunk_size=100)
    assert all(len(c.text) <= 100 for c in chunks)
    assert {(c.page, c.para) for c in chunks} == {(0, 0), (0, 1)}
    # No text lost, no overlap duplication
    rebuilt = " ".join(c.text for c in chunks if c.para == 0)
    assert rebuilt.count("جملہ") == 40


def test_chunks_split_on_urdu_full_stop():
    chunks = t.build_chunks([["الف۔ ب۔ ج۔"]], chunk_size=5)
    assert len(chunks) >= 3
    assert all(c.text.endswith("۔") for c in chunks)


# ---- assembly ------------------------------------------------------------
def test_assemble_markdown_pages_and_paragraphs():
    chunks = [t.Chunk(0, 0, "a"), t.Chunk(0, 0, "b"), t.Chunk(0, 1, "c"), t.Chunk(2, 0, "d")]
    md = t.assemble_markdown(3, 1, chunks, ["A1", "A2", "C", "D"], "x.pdf")
    assert md.startswith("# Translation of x.pdf")
    assert "A1 A2\n\nC" in md
    assert md.count("\n---\n") == 2  # between 3 pages
    assert "No text detected" in md  # empty page 2
    assert md.index("A1") < md.index("C") < md.index("D")


def test_assemble_page_numbers_offset():
    md = t.assemble_markdown(1, 5, [t.Chunk(0, 0, "a")], ["A"], "x.pdf")
    assert "## Page 5 of x.pdf" in md


# ---- translation loop (stub tokenizer/model) --------------------------------
class FakeTok:
    def convert_tokens_to_ids(self, tok):
        assert tok == t.TGT_LANG
        return 7


class FakeEnc(dict):
    def to(self, dev):
        return self


class FakeTok2(FakeTok):
    def __init__(self):
        self.seen = []

    def __call__(self, texts, **kw):
        import torch
        self.seen.append(list(texts))
        return FakeEnc(input_ids=torch.ones((len(texts), 3), dtype=torch.long))

    def batch_decode(self, gen, skip_special_tokens=True):
        # one output per row; gen row 0 carries the index of the input text
        return [f"T{int(r[0])}" for r in gen]


class FakeModel:
    def __init__(self, tok):
        self.tok = tok
        self.kwargs = None

    def generate(self, input_ids, **kw):
        import torch
        self.kwargs = kw
        # map each input back to its position in the most recent batch
        batch = self.tok.seen[-1]
        return torch.tensor([[int(b[2])] for b in batch])


def test_translate_chunks_preserves_order_and_forces_english():
    tok = FakeTok2()
    model = FakeModel(tok)
    # varied lengths so length-sorting reorders them
    chunks = [t.Chunk(0, i, f"x_{i}" + "ا" * (10 - i)) for i in range(5)]
    out = t.translate_chunks(chunks, tok, model, batch_size=2, max_input_tokens=64, num_beams=1)
    assert out == [f"T{i}" for i in range(5)]
    assert model.kwargs["forced_bos_token_id"] == 7


# ---- end-to-end CLI with everything heavy mocked ------------------------------
def test_main_end_to_end(tmp_path, monkeypatch):
    pdf = tmp_path / "in.pdf"
    pdf.write_bytes(b"%PDF-1.4 stub")
    out = tmp_path / "out.md"
    ocr_cache = tmp_path / "in.ocr.json"

    monkeypatch.setattr(t, "pdf_to_images", lambda *a, **k: ["img1", "img2"])
    monkeypatch.setattr(t, "ocr_pdf", lambda imgs, *a: [["پیراگراف ایک"], ["پیراگراف دو"]])
    monkeypatch.setattr(t, "load_translator", lambda fp16: (None, None))
    monkeypatch.setattr(
        t, "translate_chunks",
        lambda chunks, tok, model, bs, mt, nb: [f"EN{i}" for i in range(len(chunks))],
    )
    monkeypatch.setattr(
        sys, "argv", ["prog", str(pdf), "-o", str(out), "--save-ocr"])
    t.main()

    md = out.read_text(encoding="utf-8")
    assert "EN0" in md and "EN1" in md and "\n---\n" in md
    assert "## Page 1 of in.pdf" in md and "## Page 2 of in.pdf" in md
    assert json.loads(ocr_cache.read_text(encoding="utf-8")) == [["پیراگراف ایک"], ["پیراگراف دو"]]

    # second run reuses the OCR cache (ocr_pdf must not be called)
    monkeypatch.setattr(t, "ocr_pdf", lambda *a, **k: pytest.fail("OCR should be cached"))
    t.main()


def test_main_missing_file(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["prog", str(tmp_path / "nope.pdf")])
    with pytest.raises(SystemExit):
        t.main()


def test_default_output_name_matches_pdf(tmp_path, monkeypatch):
    pdf = tmp_path / "My Book.pdf"
    pdf.write_bytes(b"x")
    _stub_heavy(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["prog", str(pdf)])
    t.main()
    assert (tmp_path / "My Book.en.md").exists()


def test_directory_input_with_output_dir(tmp_path, monkeypatch):
    src, dst = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    for n in ("a.pdf", "b.PDF"):
        (src / n).write_bytes(b"x")
    (src / "notes.txt").write_text("ignore")
    loads = []
    _stub_heavy(monkeypatch, loads)
    monkeypatch.setattr(sys, "argv", ["prog", str(src), "--output-dir", str(dst)])
    t.main()
    assert sorted(p.name for p in dst.glob("*.md")) == ["a.en.md", "b.en.md"]
    assert len(list(dst.glob("urdu_translate-*.log"))) == 1
    assert len(loads) == 1  # model loaded once for the whole batch


def _stub_heavy(monkeypatch, loads=None):
    monkeypatch.setattr(t, "pdf_to_images", lambda *a, **k: ["img"])
    monkeypatch.setattr(t, "ocr_pdf", lambda imgs, *a: [["متن"]])
    monkeypatch.setattr(t, "load_translator", lambda fp16: (loads.append(1) if loads is not None else None, None))
    monkeypatch.setattr(t, "translate_chunks", lambda c, *a: ["EN"] * len(c))


class _FakePred:
    def __init__(self, lines):
        self.text_lines = lines


def test_ocr_logs_each_page_to_stdout_with_real_page_numbers(monkeypatch, capsys):
    pages = [[line("متن", 0, 20)], [], [line("اور", 0, 20)]]
    monkeypatch.setattr(t, "load_ocr", lambda: (
        lambda imgs, det_predictor: [_FakePred(pages.pop(0)) for _ in imgs], None))
    result = t.ocr_pdf(["a", "b", "c"], ocr_batch=2, first_page=10)
    cap = capsys.readouterr()
    assert result == [["متن"], [], ["اور"]]
    for n in (10, 11, 12):
        assert f"OCR page {n}:" in cap.out
    assert "no text detected" in cap.out
    assert "OCR (Surya)" in cap.out          # progress bar on stdout...
    assert "OCR (Surya)" not in cap.err      # ...not stderr


def test_translation_progress_bar_on_stdout(capsys):
    tok = FakeTok2()
    t.translate_chunks([t.Chunk(0, 0, "x_0")], tok, FakeModel(tok), 1, 64, 1)
    cap = capsys.readouterr()
    assert "Translating (NLLB)" in cap.out and "Translating" not in cap.err


def test_log_file_written_with_header_and_collapsed_progress(tmp_path, monkeypatch):
    pdf = tmp_path / "in.pdf"
    pdf.write_bytes(b"x")
    log = tmp_path / "run.log"
    _stub_heavy(monkeypatch)
    monkeypatch.setattr(t, "ocr_pdf", lambda *a: _noisy_ocr())
    monkeypatch.setattr(sys, "argv", ["prog", str(pdf), "--log-file", str(log)])
    t.main()
    text = log.read_text(encoding="utf-8")
    assert "command:" in text and "packages:" in text and "device:" in text
    assert "=== in.pdf ===" in text and "Wrote" in text
    assert "Summary: 1 succeeded, 0 failed" in text
    assert "[err] some warning" in text
    assert "\r" not in text
    assert text.count("bar 100%") == 1 and "bar 10%" not in text   # redraws collapsed
    assert not isinstance(sys.stdout, t._Tee)


def _noisy_ocr():
    sys.stderr.write("some warning\n")
    sys.stdout.write("bar 10%\rbar 50%\rbar 100%\n")
    return [["متن"]]


def test_failure_is_logged_and_batch_continues(tmp_path, monkeypatch):
    src = tmp_path / "in"
    src.mkdir()
    for n in ("a.pdf", "b.pdf"):
        (src / n).write_bytes(b"x")
    log = tmp_path / "run.log"
    _stub_heavy(monkeypatch)

    def ocr(imgs, *a):
        if monkeypatch_state["n"] == 0:
            monkeypatch_state["n"] += 1
            raise RuntimeError("MPS exploded")
        return [["متن"]]

    monkeypatch_state = {"n": 0}
    monkeypatch.setattr(t, "ocr_pdf", ocr)
    monkeypatch.setattr(sys, "argv", ["prog", str(src), "--output-dir", str(tmp_path / "o"),
                                      "--log-file", str(log)])
    with pytest.raises(SystemExit) as e:
        t.main()
    assert e.value.code == 1
    text = log.read_text(encoding="utf-8")
    assert "FAILED a.pdf" in text and "RuntimeError: MPS exploded" in text and "Traceback" in text
    assert "Summary: 1 succeeded, 1 failed: a.pdf" in text
    assert (tmp_path / "o" / "b.en.md").exists()           # second PDF still processed
    assert not isinstance(sys.stdout, t._Tee)               # streams restored


def test_default_log_location(tmp_path):
    a = SimpleNamespace(output_dir=None, pdf=tmp_path)
    assert t.default_log_path(a).parent == tmp_path
    a = SimpleNamespace(output_dir=tmp_path / "o", pdf=tmp_path)
    assert t.default_log_path(a).parent == tmp_path / "o"
    assert t.default_log_path(a).name.startswith("urdu_translate-")


def test_page_count_logged_and_mismatch_warned(tmp_path, monkeypatch, capsys):
    pdf = tmp_path / "in.pdf"
    pdf.write_bytes(b"x")
    _stub_heavy(monkeypatch)
    monkeypatch.setattr(t, "pdf_page_count", lambda p: 76)   # stub renders only 1 page
    monkeypatch.setattr(sys, "argv", ["prog", str(pdf), "--log-file", str(tmp_path / "l.log")])
    t.main()
    cap = capsys.readouterr()
    assert "PDF has 76 pages" in cap.out
    assert "expected 76 pages but rendered 1" in cap.err
    assert "PDF has 76 pages" in (tmp_path / "l.log").read_text(encoding="utf-8")


def test_page_range_logged(tmp_path, monkeypatch, capsys):
    pdf = tmp_path / "in.pdf"
    pdf.write_bytes(b"x")
    _stub_heavy(monkeypatch)
    monkeypatch.setattr(t, "pdf_page_count", lambda p: 76)
    monkeypatch.setattr(sys, "argv", ["prog", str(pdf), "--first-page", "1", "--last-page", "1",
                                      "--log-file", str(tmp_path / "l.log")])
    t.main()
    assert "processing pages 1-1" in capsys.readouterr().out


def test_page_count_unreadable_is_not_fatal(tmp_path, capsys):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"not a pdf")
    assert t.pdf_page_count(bad) is None
    assert "Could not read page count" in capsys.readouterr().err


def test_markdown_to_pdf_real_render(tmp_path):
    pytest.importorskip("weasyprint")
    md = tmp_path / "b.en.md"
    md.write_text(t.assemble_markdown(
        2, 1, [t.Chunk(0, 0, "a"), t.Chunk(1, 0, "b")],
        ["First page <b>text</b> & ā ṣ “quotes” — dash", "Second page"], "b.pdf"), encoding="utf-8")
    out = tmp_path / "b.en.pdf"
    t.markdown_to_pdf(md, out)
    data = out.read_bytes()
    assert data.startswith(b"%PDF")
    pages = t.pdf_page_count(out)       # needs poppler; None if unavailable
    if pages is not None:
        assert pages == 2               # '---' between source pages => separate PDF pages


def test_to_pdf_flag_writes_pdf_next_to_md(tmp_path, monkeypatch):
    pdf = tmp_path / "in.pdf"
    pdf.write_bytes(b"x")
    _stub_heavy(monkeypatch)
    calls = []
    monkeypatch.setattr(t, "markdown_to_pdf", lambda md, out: calls.append((md.name, out.name)))
    monkeypatch.setattr(sys, "argv", ["prog", str(pdf), "--to-pdf", "--log-file", str(tmp_path / "l.log")])
    t.main()
    assert calls == [("in.en.md", "in.en.pdf")]
