"""Page preparation before OCR: strip scanner borders and split two-page spreads.

Pure Pillow (no torch/surya), so it is cheap to test. Scans of bound books are often
one image per *spread*: two facing pages with black scanner borders and a dark gutter.
OCR run on the whole image reads across both pages and mixes the columns.

Approach: threshold to "paper" vs "dark", take the fraction of paper pixels per column,
and treat each wide run of paper columns as one page. Rows are trimmed the same way.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

PAPER_LEVEL = 128        # grey level above which a pixel counts as paper
MIN_PAPER_FRAC = 0.35    # a column/row is "paper" if at least this fraction is bright
MIN_PAGE_WIDTH = 0.15    # a run of paper columns narrower than this (x image width) is noise
MAX_GAP = 0.012          # dark gaps narrower than this (x width) don't separate pages
_WORK_WIDTH = 400        # analysis runs on a downscaled copy


@dataclass
class PreparedPage:
    image: Image.Image
    sheet: int           # 1-based index of the source image (PDF page)
    part: int            # 1-based position within the sheet, in reading order
    parts: int           # how many pages were found on the sheet
    label: str           # human-readable, e.g. "5 (right page)"
    rotated: bool = False  # the scan was sideways and was turned upright


def _paper_profile(mask: Image.Image, axis: str) -> list[float]:
    """Mean paper fraction per column (axis='x') or row (axis='y') of a 0/255 mask."""
    w, h = mask.size
    small = mask.resize((w, 1) if axis == "x" else (1, h), Image.BOX)
    return [v / 255 for v in small.getdata()]


def _runs(flags: list[bool], min_len: int, max_gap: int) -> list[tuple[int, int]]:
    """(start, end_exclusive) runs of True; gaps of <= max_gap False values are bridged."""
    runs: list[list[int]] = []
    for i, f in enumerate(flags):
        if not f:
            continue
        if runs and i - runs[-1][1] <= max_gap + 1:
            runs[-1][1] = i
        else:
            runs.append([i, i])
    return [(a, b + 1) for a, b in runs if b + 1 - a >= min_len]


def find_page_boxes(img: Image.Image) -> list[tuple[int, int, int, int]]:
    """Bounding boxes (full-resolution pixels) of the pages on a scan, left to right."""
    gray = img.convert("L")
    W, H = gray.size
    scale = min(1.0, _WORK_WIDTH / W)
    small = gray.resize((max(1, round(W * scale)), max(1, round(H * scale))), Image.BOX)
    mask = small.point(lambda v: 255 if v > PAPER_LEVEL else 0)
    sw, sh = small.size

    cols = _paper_profile(mask, "x")
    segs = _runs([c >= MIN_PAPER_FRAC for c in cols],
                 min_len=max(3, int(MIN_PAGE_WIDTH * sw)), max_gap=max(1, int(MAX_GAP * sw)))
    boxes = []
    for x0, x1 in segs:
        rows = _paper_profile(mask.crop((x0, 0, x1, sh)), "y")
        rruns = _runs([r >= MIN_PAPER_FRAC for r in rows], min_len=max(3, sh // 5),
                      max_gap=max(1, int(0.03 * sh)))
        if not rruns:
            continue
        y0, y1 = rruns[0][0], rruns[-1][1]
        inv = 1 / scale
        boxes.append((round(x0 * inv), round(y0 * inv), min(W, round(x1 * inv)), min(H, round(y1 * inv))))
    return boxes


def _turn(img: Image.Image, direction: str) -> Image.Image:
    return img.rotate(-90 if direction == "cw" else 90, expand=True)


def is_sideways(boxes, rboxes) -> bool:
    """Layout test for a scan that was fed in turned 90 degrees. Pages are portrait and
    spreads are two side-by-side, so a scan is sideways if a single page box is wider than
    tall, or if two pages only show up side by side after turning it."""
    if len(boxes) == 1:
        x0, y0, x1, y1 = boxes[0]
        if (x1 - x0) > 1.1 * (y1 - y0):
            return True
    if len(boxes) <= 1 and len(rboxes) == 2:
        w = sorted(b[2] - b[0] for b in rboxes)
        return w[0] >= 0.75 * w[1]  # facing pages are about equally wide; a split card is not
    return False


def prepare_sheet(img: Image.Image, sheet: int, rtl: bool = True, trim: int = 4,
                  sideways: str = "cw") -> list[PreparedPage]:
    """Split one scan into pages and trim borders. Falls back to the whole image if
    no paper is found. `rtl` orders pages right-to-left (Urdu books). A sideways scan is
    turned upright first, `sideways` = "cw" or "ccw" (the way to turn it; use "off" to skip)."""
    boxes = find_page_boxes(img)
    rotated = False
    if sideways != "off":
        turned = _turn(img, sideways)
        rboxes = find_page_boxes(turned)
        if is_sideways(boxes, rboxes):
            img, boxes, rotated = turned, rboxes, True
    boxes = boxes or [(0, 0, img.width, img.height)]
    if rtl:
        boxes = boxes[::-1]
    names = {1: [""], 2: [" (right page)", " (left page)"] if rtl else [" (left page)", " (right page)"]}
    pages = []
    for i, (x0, y0, x1, y1) in enumerate(boxes, start=1):
        box = (x0 + trim, y0 + trim, max(x0 + trim + 1, x1 - trim), max(y0 + trim + 1, y1 - trim))
        suffix = names.get(len(boxes), [f" (part {j})" for j in range(1, len(boxes) + 1)])[i - 1]
        pages.append(PreparedPage(img.crop(box), sheet, i, len(boxes), f"{sheet}{suffix}", rotated))
    return pages


def prepare_pages(images, first_sheet: int = 1, rtl: bool = True,
                  save_dir: Path | None = None, sideways: str = "cw") -> list[PreparedPage]:
    """prepare_sheet over every scan, optionally saving each page as a PNG for inspection."""
    out: list[PreparedPage] = []
    for n, img in enumerate(images, start=first_sheet):
        out.extend(prepare_sheet(img, n, rtl, sideways=sideways))
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        for p in out:
            p.image.save(save_dir / f"sheet{p.sheet:03d}-{p.part}.png")
    return out
