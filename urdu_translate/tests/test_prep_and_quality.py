"""Page preparation (real Pillow images) and hallucination guards. No models needed."""
from types import SimpleNamespace

from PIL import Image, ImageDraw

import page_prep as pp
import text_quality as tq
import translate_urdu_pdf as t


def spread(w=800, h=600):
    """Black scan with two white pages split by a dark gutter, 'text' on each."""
    im = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(im)
    d.rectangle((40, 30, 380, 570), fill=255)     # left page
    d.rectangle((420, 30, 770, 570), fill=255)    # right page
    for x0 in (60, 440):
        for y in range(60, 540, 40):
            d.rectangle((x0, y, x0 + 280, y + 8), fill=0)
    return im


# ---- page prep --------------------------------------------------------------
def test_spread_split_right_to_left_and_borders_cropped():
    pages = pp.prepare_sheet(spread(), sheet=7)
    assert [p.label for p in pages] == ["7 (right page)", "7 (left page)"]
    right, left = pages
    assert all(p.parts == 2 for p in pages)
    # each crop is roughly one page wide and has no black border left
    for p in pages:
        assert 300 <= p.image.width <= 360 and 500 <= p.image.height <= 545
        assert p.image.getpixel((2, 2)) > 200
    # right page was the second box in the scan
    assert right.image.size == pp.prepare_sheet(spread(), 1)[0].image.size


def test_ltr_order():
    assert [p.label for p in pp.prepare_sheet(spread(), 3, rtl=False)] == [
        "3 (left page)", "3 (right page)"]


def test_single_page_only_gets_border_crop():
    im = Image.new("L", (500, 700), 0)
    ImageDraw.Draw(im).rectangle((50, 80, 450, 650), fill=255)
    (p,) = pp.prepare_sheet(im, 1)
    assert p.label == "1" and p.parts == 1
    assert 380 <= p.image.width <= 400 and 550 <= p.image.height <= 570


def test_blank_black_image_falls_back_to_whole_image():
    (p,) = pp.prepare_sheet(Image.new("L", (300, 400), 0), 1)
    assert p.image.size[0] > 280


def test_prepare_pages_numbers_and_saves(tmp_path):
    pages = pp.prepare_pages([spread(), spread()], first_sheet=10, save_dir=tmp_path / "pg")
    assert [p.sheet for p in pages] == [10, 10, 11, 11]
    assert sorted(f.name for f in (tmp_path / "pg").iterdir()) == [
        "sheet010-1.png", "sheet010-2.png", "sheet011-1.png", "sheet011-2.png"]


# ---- text quality -----------------------------------------------------------
def test_degenerate_detects_loops_and_dominant_words():
    assert tq.is_degenerate("nom nom nom nom nom nom nom nom")
    assert tq.is_degenerate("المراهم المراهم المراهم المراهم المراهم المراهم المراهم")
    assert tq.is_degenerate("The state of the state of the state of the state of the state of the state")
    assert tq.is_degenerate("nomnomnomnomnomnomnom is nom")


def test_normal_prose_not_degenerate():
    assert not tq.is_degenerate("This catalog lists scientific books printed in Urdu between 1850 and 1940.")
    assert not tq.is_degenerate("کتاب")  # too short to judge


def test_illegible_source_flags_cjk_noise():
    assert tq.is_illegible_source("できるxもうかししにいかしないという لاده")
    assert not tq.is_illegible_source("اردو میں سائنسی ادب خواجہ حمید الدین شاہد")


def test_translation_unreliable_length_and_loops():
    src = "یہ ایک سائنسی کتاب ہے جو لاہور میں چھپی"
    assert not tq.translation_unreliable(src, "This is a scientific book printed in Lahore")
    assert tq.translation_unreliable(src, "nom nom nom nom nom nom nom")
    assert tq.translation_unreliable(src, " ".join(["Once upon a time a king lived far away"] * 5))
    assert tq.translation_unreliable(src, "")


# ---- integration with translate_chunks ----------------------------------------
class _Tok:
    def convert_tokens_to_ids(self, tok):
        return 7

    def __call__(self, texts, **kw):
        import torch

        class Enc(dict):
            def to(self, dev):
                return self
        self.texts = list(texts)
        return Enc(input_ids=torch.ones((len(texts), 3), dtype=torch.long))

    def batch_decode(self, gen, skip_special_tokens=True):
        return [self.replies[tx] for tx in self.texts]


class _Model:
    def generate(self, input_ids, **kw):
        import torch
        return torch.zeros((input_ids.shape[0], 2), dtype=torch.long)


def test_translate_chunks_skips_illegible_and_flags_hallucination(capsys):
    good = "یہ ایک سائنسی کتاب ہے جو لاہور میں چھپی"
    loop = "المراهم " * 12
    bad_src = "ایک دو تین چار پانچ چھ سات آٹھ"
    tok = _Tok()
    tok.replies = {good: "This is a scientific book printed in Lahore",
                   bad_src: "nom nom nom nom nom nom"}
    chunks = [t.Chunk(0, 0, good), t.Chunk(0, 1, loop.strip()), t.Chunk(0, 2, bad_src)]
    out = t.translate_chunks(chunks, tok, _Model(), batch_size=4, max_input_tokens=64, num_beams=1)
    assert out == ["This is a scientific book printed in Lahore", tq.ILLEGIBLE, tq.UNRELIABLE]
    assert loop.strip() not in tok.texts          # garbage never reached the model
    assert "1 of 3 chunks skipped as illegible OCR, 1 translations flagged" in capsys.readouterr().out


def test_assemble_uses_labels_and_marks_placeholders():
    chunks = [t.Chunk(0, 0, "a"), t.Chunk(1, 0, "b")]
    md = t.assemble_markdown(2, 1, chunks, [tq.ILLEGIBLE, "Hello"], "x.pdf",
                             labels=["5 (right page)", "5 (left page)"])
    assert "## Page 5 (right page) of x.pdf" in md and "## Page 5 (left page) of x.pdf" in md
    assert "*[illegible]*" in md


# ---- OCR checkpoint / resume ---------------------------------------------------
def test_ocr_checkpoint_resumes_and_retries_oom(tmp_path, monkeypatch):
    import json
    calls = []

    class Pred:
        def __init__(self, txt):
            self.text_lines = [SimpleNamespace(text=txt, bbox=(0, 0, 10, 10))]

    state = {"boom": True}

    def rec(imgs, det_predictor):
        calls.append(list(imgs))
        if imgs == ["c"] and state["boom"]:
            state["boom"] = False
            raise RuntimeError("MPS backend out of memory")
        return [Pred(f"txt-{i}") for i in imgs]

    monkeypatch.setattr(t, "load_ocr", lambda: (rec, None))
    ck = tmp_path / "x.partial.json"
    ck.write_text(json.dumps([["txt-a"]]))            # page 1 already done
    pages = t.ocr_pdf(["a", "b", "c"], ocr_batch=1, checkpoint=ck)
    assert pages == [["txt-a"], ["txt-b"], ["txt-c"]]
    assert calls[0] == ["b"]                           # did not redo page 1
    assert calls.count(["c"]) == 2                     # OOM retried once
    assert json.loads(ck.read_text()) == pages         # checkpoint kept up to date
