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
from tqdm import tqdm  # noqa: E402

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


def ocr_pdf(images, ocr_batch: int) -> list[list[str]]:
    """Returns pages -> paragraphs. Processes a few pages at a time to bound memory."""
    rec, det = load_ocr()
    pages: list[list[str]] = []
    for i in tqdm(range(0, len(images), ocr_batch), desc="OCR (Surya)", unit="batch"):
        batch = images[i:i + ocr_batch]
        preds = rec(batch, det_predictor=det)
        for p in preds:
            pages.append(lines_to_paragraphs(p.text_lines))
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
        parts.append(f"<!-- Page {first_page + p} -->")
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
    ap.add_argument("pdf", type=Path)
    ap.add_argument("-o", "--output", type=Path, help="Output .md (default: <pdf>.en.md)")
    ap.add_argument("--dpi", type=int, default=200, help="Render DPI (default 200)")
    ap.add_argument("--first-page", type=int, help="1-based first page")
    ap.add_argument("--last-page", type=int, help="1-based last page")
    ap.add_argument("--ocr-batch", type=int, default=2, help="Pages per OCR batch")
    ap.add_argument("--chunk-size", type=int, default=400, help="Max chars per chunk")
    ap.add_argument("--batch-size", type=int, default=4, help="Chunks per translation batch")
    ap.add_argument("--beams", type=int, default=4)
    ap.add_argument("--max-input-tokens", type=int, default=256)
    ap.add_argument("--fp16", action="store_true", help="float16 model (less RAM; may be less stable on MPS)")
    ap.add_argument("--save-ocr", type=Path, help="Cache OCR text as JSON here (reused if it exists)")
    return ap.parse_args()


def main():
    a = parse_args()
    if not a.pdf.is_file():
        sys.exit(f"File not found: {a.pdf}")
    out = a.output or a.pdf.with_suffix(".en.md")
    print(f"Device: {DEVICE}")

    if a.save_ocr and a.save_ocr.exists():
        pages = json.loads(a.save_ocr.read_text(encoding="utf-8"))
        print(f"Loaded cached OCR: {len(pages)} pages")
    else:
        images = pdf_to_images(a.pdf, a.dpi, a.first_page, a.last_page)
        print(f"Rendered {len(images)} pages at {a.dpi} DPI")
        pages = ocr_pdf(images, a.ocr_batch)
        del images
        if a.save_ocr:
            a.save_ocr.write_text(json.dumps(pages, ensure_ascii=False, indent=1), encoding="utf-8")

    chunks = build_chunks(pages, a.chunk_size)
    print(f"{len(chunks)} chunks from {len(pages)} pages")

    tok, model = load_translator(a.fp16)
    translations = translate_chunks(chunks, tok, model, a.batch_size, a.max_input_tokens, a.beams)

    md = assemble_markdown(len(pages), a.first_page or 1, chunks, translations, a.pdf.name)
    out.write_text(md, encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
