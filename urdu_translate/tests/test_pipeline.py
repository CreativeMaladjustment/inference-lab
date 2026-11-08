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
    monkeypatch.setattr(t, "ocr_pdf", lambda imgs, b: [["پیراگراف ایک"], ["پیراگراف دو"]])
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
    monkeypatch.setattr(t, "ocr_pdf", lambda *a: pytest.fail("OCR should be cached"))
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
    assert sorted(p.name for p in dst.iterdir()) == ["a.en.md", "b.en.md"]
    assert len(loads) == 1  # model loaded once for the whole batch


def _stub_heavy(monkeypatch, loads=None):
    monkeypatch.setattr(t, "pdf_to_images", lambda *a, **k: ["img"])
    monkeypatch.setattr(t, "ocr_pdf", lambda imgs, b: [["متن"]])
    monkeypatch.setattr(t, "load_translator", lambda fp16: (loads.append(1) if loads is not None else None, None))
    monkeypatch.setattr(t, "translate_chunks", lambda c, *a: ["EN"] * len(c))
