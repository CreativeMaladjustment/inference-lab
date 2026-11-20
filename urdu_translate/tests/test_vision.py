"""Vision step with a fake provider: no network, no SDK."""
import json

import pytest
from PIL import Image

import vision_translate as v


def png(path, size=(60, 80)):
    Image.new("L", size, 255).save(path)


def ok(text, leg="clear", page="", notes=""):
    return json.dumps({"legibility": leg, "printed_page_number": page,
                       "translation": text, "notes": notes})


class Fake:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def translate(self, image, mime, temperature=0.0, note=""):
        self.calls.append(temperature)
        self.notes = getattr(self, "notes", []) + [note]
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


NOSLEEP = lambda s: None


def test_parse_result_validates():
    assert v.parse_result(ok("Hello", page=5))["printed_page_number"] == "5"
    for bad in ("not json", "[]", json.dumps({"legibility": "great"}), ""):
        with pytest.raises(ValueError):
            v.parse_result(bad)


def test_looks_stuck_ignores_ditto_marks_and_flags_loops():
    table = "9. Optics 1840 〃 〃 〃 〃 〃 10. Electricity 1841 〃 〃 〃 〃 〃 11. Globes 1842"
    assert not v.looks_stuck(table)
    assert v.looks_stuck("the book the book the book the book the book the book the book")
    assert not v.looks_stuck("Usul-e-Ilm-e-Hisab (Principles of Arithmetic), printed 1836.")


def test_translate_page_retries_api_error_then_succeeds(tmp_path):
    p = tmp_path / "sheet001-1.png"; png(p)
    fake = Fake([RuntimeError("429 rate limited"), "garbage", ok("A catalog entry.")])
    r = v.translate_page(fake, p, 2000, sleep=NOSLEEP)
    assert r["translation"] == "A catalog entry." and len(fake.calls) == 3


def test_translate_page_gives_up_after_retries(tmp_path):
    p = tmp_path / "sheet001-1.png"; png(p)
    fake = Fake([RuntimeError("boom")] * 5)
    with pytest.raises(RuntimeError):
        v.translate_page(fake, p, 2000, retries=4, sleep=NOSLEEP)


def test_stuck_translation_retried_warmer_then_flagged(tmp_path):
    p = tmp_path / "sheet001-1.png"; png(p)
    loop = ok("nom nom nom nom nom nom nom nom")
    fake = Fake([loop, loop])
    r = v.translate_page(fake, p, 2000, sleep=NOSLEEP)
    assert fake.calls == [0.0, 0.4] and r["flag"] == "stuck"
    fake2 = Fake([loop, ok("Real text here.")])
    assert v.translate_page(fake2, p, 2000, sleep=NOSLEEP)["translation"] == "Real text here."


def test_render_page_variants():
    r = {"legibility": "clear", "printed_page_number": "46", "translation": "Text.", "notes": ""}
    assert "## Page 5 (printed page 46)" in v.render_page("5", r)
    assert "*[illegible]*" in v.render_page("6", {**r, "legibility": "illegible", "translation": ""})
    assert "Partly legible" in v.render_page("7", {**r, "legibility": "partial"})
    assert "> handwritten" in v.render_page("8", {**r, "notes": "handwritten"})
    assert v.UNRELIABLE in v.render_page("9", {**r, "flag": "stuck"})
    assert "not translated: boom" in v.render_page("10", None, "boom")


def test_run_writes_markdown_resumes_from_cache_and_records_failures(tmp_path):
    pages = tmp_path / "book.pages"; pages.mkdir()
    files = []
    for n in (1, 2, 3):
        f = pages / f"sheet00{n}-1.png"; png(f); files.append(f)
    out = tmp_path / "out.md"
    labels = ["1", "2 (right page)", "3"]
    fake = Fake([ok("One."), RuntimeError("down")] + [RuntimeError("down")] * 4 + [ok("Three.")])
    failed = v.run(files, labels, fake, out, max_side=2000, delay=0, source_name="book", sleep=NOSLEEP)
    assert failed == ["sheet002-1.png"]
    md = out.read_text()
    assert "One." in md and "Three." in md and "not translated" in md
    assert md.index("## Page 1") < md.index("## Page 2 (right page)") < md.index("## Page 3")
    # second run: only the failed page is retried; finished pages cost nothing
    fake2 = Fake([ok("Two.")])
    assert v.run(files, labels, fake2, out, max_side=2000, delay=0, source_name="book", sleep=NOSLEEP) == []
    assert len(fake2.calls) == 1 and "Two." in out.read_text()


def test_labels_from_json_when_counts_match(tmp_path):
    files = [tmp_path / "sheet001-1.png", tmp_path / "sheet002-1.png"]
    lj = tmp_path / "l.json"; lj.write_text(json.dumps(["1", "2 (right page)"]))
    assert v.load_labels(files, lj) == ["1", "2 (right page)"]
    lj.write_text(json.dumps(["only one"]))
    assert v.load_labels(files, lj) == ["1 (part 1)", "2 (part 1)"]


def test_main_dry_run_and_requirements(tmp_path, capsys, monkeypatch):
    pages = tmp_path / "b.pages"; pages.mkdir()
    for n in (1, 2, 3):
        png(pages / f"sheet00{n}-1.png")
    v.main([str(pages), "--dry-run", "--limit", "2"])
    assert "Would translate 2 page images" in capsys.readouterr().out
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    with pytest.raises(SystemExit, match="--model"):
        v.main([str(pages)])
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="GEMINI_API_KEY"):
        v.main([str(pages), "--model", "m"])


# ---- local (mlx) provider: tagged answer -> JSON --------------------------------------
def test_tagged_answer_converted():
    text = ("LEGIBILITY: clear\nPRINTED_PAGE: 46\nNOTES:\nTRANSLATION:\n"
            "1. Principles of Arithmetic, printed 1836.\n2. Optics, 1840.")
    r = v.parse_result(v.tagged_to_json(text))
    assert r["legibility"] == "clear" and r["printed_page_number"] == "46" and r["notes"] == ""
    assert r["translation"] == "1. Principles of Arithmetic, printed 1836.\n2. Optics, 1840."


def test_tagged_answer_with_markdown_noise_and_inline_translation():
    r = v.parse_result(v.tagged_to_json("**LEGIBILITY:** Partial.\n**TRANSLATION:** Hello [illegible] world"))
    assert r["legibility"] == "partial" and r["translation"] == "Hello [illegible] world"


def test_translation_text_containing_a_tag_word_is_not_split():
    r = v.parse_result(v.tagged_to_json(
        "LEGIBILITY: clear\nTRANSLATION:\nPrinted notes: see footnote\nNOTES: 3 volumes"))
    assert "NOTES: 3 volumes" in r["translation"] and r["notes"] == ""


def test_untagged_answer_kept_as_partial_translation():
    r = v.parse_result(v.tagged_to_json("The page lists some books."))
    assert r["legibility"] == "partial" and r["translation"] == "The page lists some books."
    assert "ignored" in r["notes"]


def test_mlx_provider_requires_model(tmp_path, monkeypatch):
    pages = tmp_path / "b.pages"; pages.mkdir(); png(pages / "sheet001-1.png")
    with pytest.raises(SystemExit, match="Hugging Face"):
        v.main([str(pages), "--provider", "mlx"])


# ---- tiling -------------------------------------------------------------------------
def lined_page(tmp_path, lines=12, gap=40, w=300):
    """White page with a dark text 'line' every `gap` px."""
    from PIL import ImageDraw
    h = lines * gap + gap
    im = Image.new("L", (w, h), 255)
    d = ImageDraw.Draw(im)
    for i in range(lines):
        y = gap // 2 + i * gap
        d.rectangle((20, y, w - 20, y + 14), fill=0)
    p = tmp_path / "sheet001-1.png"; im.save(p)
    return im, p


def test_tile_cuts_land_in_gaps_between_lines(tmp_path):
    im, _ = lined_page(tmp_path)
    cuts = v.tile_cuts(im, 4)
    assert cuts[0] == 0 and cuts[-1] == im.height and len(cuts) == 5
    px = im.load()
    for c in cuts[1:-1]:
        assert all(px[x, c] == 255 for x in range(20, 280))     # cut row has no ink


def test_tile_cuts_small_or_single():
    assert v.tile_cuts(Image.new("L", (100, 30), 255), 4) == [0, 30]
    assert v.tile_cuts(Image.new("L", (100, 500), 255), 1) == [0, 500]


def test_tiled_page_translates_each_strip_and_merges(tmp_path):
    _, p = lined_page(tmp_path)
    fake = Fake([ok("Top.", page="5"), ok("Middle.", leg="partial"), ok("Bottom.")])
    r = v.translate_page(fake, p, 2000, sleep=NOSLEEP, tiles=3)
    assert r["translation"] == "Top.\n\nMiddle.\n\nBottom."
    assert r["legibility"] == "partial" and r["printed_page_number"] == "5"
    assert all("horizontal strip" in n for n in fake.notes)


def test_stuck_strip_is_marked_but_rest_of_page_kept(tmp_path):
    _, p = lined_page(tmp_path)
    loop = ok("nom nom nom nom nom nom nom nom")
    fake = Fake([ok("Top."), loop, loop, ok("Bottom.")])
    r = v.translate_page(fake, p, 2000, sleep=NOSLEEP, tiles=3)
    assert r["translation"] == "Top.\n\n" + v.UNRELIABLE + "\n\nBottom."
    assert r["legibility"] == "partial" and "flag" not in r


def test_all_strips_illegible_or_stuck():
    ill = {"legibility": "illegible", "printed_page_number": "", "translation": "", "notes": ""}
    assert v.merge_tiles([ill, ill])["legibility"] == "illegible"
    stuck = {**ill, "legibility": "clear", "translation": "x", "flag": "stuck"}
    assert v.merge_tiles([stuck, stuck])["flag"] == "stuck"
