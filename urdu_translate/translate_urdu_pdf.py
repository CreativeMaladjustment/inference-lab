#!/usr/bin/env python3
"""Urdu PDF -> English Markdown: pdf2image -> Surya OCR -> chunking -> NLLB-200 -> .md

Usage: python translate_urdu_pdf.py input.pdf [-o output.md]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

# --- Device selection MUST happen before surya is imported (it reads TORCH_DEVICE). ---
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # CPU fallback for unsupported MPS ops
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        os.environ["TORCH_DEVICE"] = "mps"  # tells Surya to use Metal too
        return torch.device("mps")
    print("WARNING: MPS not available; falling back to CPU (slow).", file=sys.stderr)
    os.environ["TORCH_DEVICE"] = "cpu"
    return torch.device("cpu")


DEVICE = get_device()

from pdf2image import convert_from_path  # noqa: E402
from tqdm import tqdm as _tqdm  # noqa: E402


def tqdm(*args, **kwargs):
    """tqdm writing to stdout (its default is stderr)."""
    kwargs.setdefault("file", sys.stdout)
    return _tqdm(*args, **kwargs)

SRC_LANG, TGT_LANG = "urd_Arab", "eng_Latn"
MODEL_NAME = "facebook/nllb-200-distilled-1.3B"


@dataclass
class Chunk:
    page: int      # 0-based page index
    para: int      # paragraph index within the page
    text: str


# ----------------------------------------------------------------- 1. PDF + OCR
def pdf_to_images(pdf_path: Path, dpi: int, first: int | None, last: int | None):
    return convert_from_path(str(pdf_path), dpi=dpi, first_page=first, last_page=last, fmt="png")


def load_ocr():
    # NOTE: current Surya releases are multilingual and take no language argument;
    # Urdu is handled by the model itself. (Only Surya <=0.8 used langs=["ur"].)
    from surya.detection import DetectionPredictor
    from surya.foundation import FoundationPredictor
    from surya.recognition import RecognitionPredictor

    foundation = FoundationPredictor()
    return RecognitionPredictor(foundation), DetectionPredictor()


def lines_to_paragraphs(lines, gap_factor: float = 1.0) -> list[str]:
    """Group OCR lines into paragraphs by vertical gap (top-to-bottom reading order)."""
    items = []
    for ln in lines:
        text = (ln.text or "").strip()
        if text:
            x0, y0, x1, y1 = ln.bbox
            items.append((y0, y1, text))
    if not items:
        return []
    items.sort(key=lambda t: (t[0] + t[1]) / 2)
    heights = [b - a for a, b, _ in items if b > a]
    med_h = statistics.median(heights) if heights else 20
    paras, cur, prev_bottom = [], [items[0][2]], items[0][1]
    for y0, y1, text in items[1:]:
        if y0 - prev_bottom > gap_factor * med_h:  # larger-than-normal gap => new paragraph
            paras.append(" ".join(cur))
            cur = []
        cur.append(text)
        prev_bottom = max(prev_bottom, y1)
    paras.append(" ".join(cur))
    return paras


def ocr_pdf(images, ocr_batch: int, first_page: int = 1) -> list[list[str]]:
    """Returns pages -> paragraphs. Processes a few pages at a time to bound memory.

    Logs one line per page (via tqdm.write so the progress bar stays intact).
    """
    rec, det = load_ocr()
    pages: list[list[str]] = []
    with tqdm(total=len(images), desc="OCR (Surya)", unit="page") as bar:
        for i in range(0, len(images), ocr_batch):
            preds = rec(images[i:i + ocr_batch], det_predictor=det)
            for p in preds:
                paras = lines_to_paragraphs(p.text_lines)
                pages.append(paras)
                num = first_page + len(pages) - 1
                chars = sum(len(x) for x in paras)
                note = "" if paras else "  (no text detected)"
                _tqdm.write(f"  OCR page {num}: {len(p.text_lines)} lines, "
                            f"{len(paras)} paragraphs, {chars} chars{note}", file=sys.stdout)
                bar.update(1)
            if DEVICE.type == "mps":
                torch.mps.empty_cache()
    return pages


# ----------------------------------------------------------------- 2. Chunking
def make_splitter(chunk_size: int, overlap: int = 0):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,  # keep 0: overlap would duplicate text in the translation
        # paragraph -> line -> Urdu full stop/question mark -> Latin punct -> Arabic comma -> word
        separators=["\n\n", "\n", "۔", "؟", "!", ".", "?", "،", " ", ""],
        keep_separator="end",
    )


def build_chunks(pages: list[list[str]], chunk_size: int) -> list[Chunk]:
    splitter = make_splitter(chunk_size)
    chunks: list[Chunk] = []
    for pi, paras in enumerate(pages):
        for qi, para in enumerate(paras):
            for piece in splitter.split_text(para):
                if piece.strip():
                    chunks.append(Chunk(pi, qi, piece.strip()))
    return chunks


# ----------------------------------------------------------------- 3. Translation
def load_translator(fp16: bool):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang=SRC_LANG)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16 if fp16 else torch.float32
    ).to(DEVICE).eval()
    return tok, model


@torch.inference_mode()
def translate_chunks(chunks: list[Chunk], tok, model, batch_size: int,
                     max_input_tokens: int, num_beams: int) -> list[str]:
    out = [""] * len(chunks)
    # Sort by length so batches have little padding; restore order via index.
    order = sorted(range(len(chunks)), key=lambda i: len(chunks[i].text))
    bos = tok.convert_tokens_to_ids(TGT_LANG)
    with tqdm(total=len(chunks), desc="Translating (NLLB)", unit="chunk") as bar:
        for s in range(0, len(order), batch_size):
            idx = order[s:s + batch_size]
            enc = tok([chunks[i].text for i in idx], return_tensors="pt", padding=True,
                      truncation=True, max_length=max_input_tokens).to(DEVICE)
            gen = model.generate(
                **enc,
                forced_bos_token_id=bos,
                num_beams=num_beams,
                max_new_tokens=int(enc["input_ids"].shape[1] * 1.6) + 16,
                no_repeat_ngram_size=4,
            )
            for i, text in zip(idx, tok.batch_decode(gen, skip_special_tokens=True)):
                out[i] = text.strip()
            bar.update(len(idx))
            if DEVICE.type == "mps":
                torch.mps.empty_cache()
    return out


# ----------------------------------------------------------------- 4. Reassembly
def assemble_markdown(num_pages: int, first_page: int, chunks: list[Chunk],
                      translations: list[str], source_name: str) -> str:
    # (page, para) -> joined chunk translations, in original order
    paras: dict[tuple[int, int], list[str]] = {}
    for c, t in zip(chunks, translations):
        paras.setdefault((c.page, c.para), []).append(t)

    parts = [f"# Translation of {source_name}\n"]
    for p in range(num_pages):
        parts.append(f"## Page {first_page + p} of {source_name}")
        keys = sorted(k for k in paras if k[0] == p)
        if keys:
            parts.extend(" ".join(paras[k]) for k in keys)
        else:
            parts.append("*[No text detected on this page]*")
        if p < num_pages - 1:
            parts.append("---")
    return "\n\n".join(parts) + "\n"


# ----------------------------------------------------------------- CLI
def parse_args():
    ap = argparse.ArgumentParser(description="Translate an Urdu PDF to English Markdown.")
    ap.add_argument("pdf", type=Path, help="A PDF file, or a directory of PDFs")
    ap.add_argument("-o", "--output", type=Path,
                    help="Output .md for a single PDF (default: <pdf name>.en.md next to the PDF)")
    ap.add_argument("--output-dir", type=Path,
                    help="Write <pdf name>.en.md files here instead of next to each PDF")
    ap.add_argument("--dpi", type=int, default=200, help="Render DPI (default 200)")
    ap.add_argument("--first-page", type=int, help="1-based first page")
    ap.add_argument("--last-page", type=int, help="1-based last page")
    ap.add_argument("--ocr-batch", type=int, default=2, help="Pages per OCR batch")
    ap.add_argument("--chunk-size", type=int, default=400, help="Max chars per chunk")
    ap.add_argument("--batch-size", type=int, default=4, help="Chunks per translation batch")
    ap.add_argument("--beams", type=int, default=4)
    ap.add_argument("--max-input-tokens", type=int, default=256)
    ap.add_argument("--fp16", action="store_true", help="float16 model (less RAM; may be less stable on MPS)")
    ap.add_argument("--save-ocr", action="store_true",
                    help="Cache OCR text as <pdf name>.ocr.json beside the output (reused if present)")
    return ap.parse_args()


def find_pdfs(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.suffix.lower() == ".pdf")
    return [path]


def output_path_for(pdf: Path, a) -> Path:
    if a.output and not a.pdf.is_dir():
        return a.output
    folder = a.output_dir or pdf.parent
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{pdf.stem}.en.md"  # book.pdf -> book.en.md


def process_pdf(pdf: Path, out: Path, a, translator) -> None:
    cache = out.with_name(f"{pdf.stem}.ocr.json") if a.save_ocr else None
    if cache and cache.exists():
        pages = json.loads(cache.read_text(encoding="utf-8"))
        print(f"Loaded cached OCR: {len(pages)} pages")
    else:
        images = pdf_to_images(pdf, a.dpi, a.first_page, a.last_page)
        print(f"Rendered {len(images)} pages at {a.dpi} DPI")
        pages = ocr_pdf(images, a.ocr_batch, a.first_page or 1)
        del images
        if cache:
            cache.write_text(json.dumps(pages, ensure_ascii=False, indent=1), encoding="utf-8")

    chunks = build_chunks(pages, a.chunk_size)
    print(f"{len(chunks)} chunks from {len(pages)} pages")
    tok, model = translator()
    translations = translate_chunks(chunks, tok, model, a.batch_size, a.max_input_tokens, a.beams)
    out.write_text(
        assemble_markdown(len(pages), a.first_page or 1, chunks, translations, pdf.name),
        encoding="utf-8")
    print(f"Wrote {out}")


def main():
    a = parse_args()
    a.pdf = a.pdf.expanduser()
    if not a.pdf.exists():
        sys.exit(f"Not found: {a.pdf}")
    pdfs = find_pdfs(a.pdf)
    if not pdfs:
        sys.exit(f"No PDFs in {a.pdf}")
    print(f"Device: {DEVICE}; {len(pdfs)} PDF(s)")

    loaded = []  # load the 1.3B model once, only when first needed

    def translator():
        if not loaded:
            loaded.append(load_translator(a.fp16))
        return loaded[0]

    for pdf in pdfs:
        print(f"\n=== {pdf.name} ===")
        process_pdf(pdf, output_path_for(pdf, a), a, translator)


if __name__ == "__main__":
    main()
