#!/usr/bin/env python3
"""Page images -> English Markdown, using a vision LLM (Gemini) instead of OCR + NLLB.

Input is the folder of cropped/split page PNGs that translate_urdu_pdf.py writes with
--save-pages (<pdf name>.pages/). Each page image is sent to the model with a prompt that
says what the book is (a dry bibliography of Urdu science books), forbids invention, and
asks for [illegible] wherever the text cannot be read.

Usage:
  export GEMINI_API_KEY=...               # read from the environment, never from a file
  python vision_translate.py --list-models
  python vision_translate.py "book.pages" -o book.vision.en.md --model <model> --limit 3
  python vision_translate.py "book.pages" -o book.vision.en.md --model <model>

Results are cached page by page in <output>.jsonl, so an interrupted run resumes and
re-running does not pay twice for pages that are done. Page images are sent to Google's
API: only run this on material you are happy to send there.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

from text_quality import is_degenerate

SYSTEM_PROMPT = """\
You translate single page images from a scanned book. The book is "Urdu mein Scientific \
Literature" ("Scientific Literature in Urdu"), a descriptive bibliography and history of \
scientific books published in Urdu from roughly the 16th century to the early 20th century, \
published by Aiwan-e-Urdu, Karachi, in 1969. Its content is dry, factual non-fiction: \
catalog entries, author names, book titles, places and dates of publication (often Hijri \
years with the Gregorian year in brackets), publishers, quoted prefaces (some in Persian or \
Arabic), footnotes and tables of contents. Many pages are typeset; some are handwritten \
Nastaliq. Some pages may be sideways or upside down: read them in their correct orientation.

Rules:
- Translate only what is visible on the page. Never invent, infer, summarise or add \
background, and never write a story or commentary. If the page is a list, give a list.
- Keep book titles and personal names in transliteration, with a short English gloss in \
brackets when the meaning is clear. Keep dates exactly as printed and keep any Hijri/Gregorian \
pairing.
- Keep the page's structure: headings, numbered entries, footnotes, table rows.
- Translate Persian or Arabic passages too, marking them [Persian] or [Arabic].
- Where a word or passage cannot be read with confidence, write [illegible] in its place. \
For a partly readable word write [unclear: best reading]. Never fill a gap with a plausible guess.
- If you notice yourself repeating the same word or phrase, stop and write [illegible].
- If the page is blank or has no text, return an empty translation.

"""

JSON_FORMAT = """\
Return JSON with these fields:
  legibility: "clear" (typeset and readable), "partial" (some parts unreadable) or \
"illegible" (cannot be read; leave translation empty).
  printed_page_number: the page number printed on the page, or "" if none is visible.
  translation: the English translation, as plain text.
  notes: anything the reader should know about this page (for example "handwritten"), or "".
"""

# Local models can't be forced into a JSON schema, and quotes/newlines inside a JSON string
# are fragile for them, so they answer in tagged plain text that tagged_to_json() converts.
TAGGED_FORMAT = """\
Answer in exactly this format and nothing else:
LEGIBILITY: clear, partial or illegible (clear = typeset and readable; illegible = cannot be \
read, leave the translation empty)
PRINTED_PAGE: the page number printed on the page, or leave empty
NOTES: anything the reader should know (for example "handwritten"), or leave empty
TRANSLATION:
the English translation as plain text, starting on the next line
"""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "legibility": {"type": "string", "enum": ["clear", "partial", "illegible"]},
        "printed_page_number": {"type": "string"},
        "translation": {"type": "string"},
        "notes": {"type": "string"},
    },
    "required": ["legibility", "printed_page_number", "translation", "notes"],
}

TILE_NOTE = ("This image is one horizontal strip cut from a larger page, so it may start or end "
             "partway through a paragraph or table. Translate only what is visible in the strip; "
             "do not comment on what is missing, and do not repeat anything.")

UNRELIABLE = "[translation unreliable]"
ILLEGIBLE = "[illegible]"


# ----------------------------------------------------------------- provider
class GeminiProvider:
    """Thin wrapper over the google-genai SDK (imported lazily so tests don't need it)."""

    def __init__(self, model: str, api_key: str):
        from google import genai

        self.model, self._client = model, genai.Client(api_key=api_key)

    def translate(self, image: bytes, mime: str, temperature: float = 0.0, note: str = "") -> str:
        from google.genai import types

        resp = self._client.models.generate_content(
            model=self.model,
            contents=[types.Part.from_bytes(data=image, mime_type=mime),
                      ("Translate this page following the rules. " + note).strip()],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT + JSON_FORMAT,
                temperature=temperature,
                response_mime_type="application/json",
                response_json_schema=RESPONSE_SCHEMA,
            ),
        )
        return resp.text or ""

    @staticmethod
    def list_models(api_key: str) -> list[str]:
        from google import genai

        out = []
        for m in genai.Client(api_key=api_key).models.list():
            if "generateContent" in (getattr(m, "supported_actions", None) or ["generateContent"]):
                out.append(m.name)
        return sorted(out)


_TAG = re.compile(r"^\s*\**(LEGIBILITY|PRINTED_PAGE|NOTES|TRANSLATION)\**\s*:\s*\**\s*(.*)$", re.I)


def tagged_to_json(text: str) -> str:
    """Convert a local model's tagged-text answer into the JSON string parse_result expects.
    An answer that ignored the format is kept as the translation, marked partial, with a note."""
    fields: dict[str, str] = {}
    current = None
    for line in text.strip().splitlines():
        m = _TAG.match(line)
        if m and (m.group(1).upper() == "TRANSLATION" or current != "TRANSLATION"):
            current = m.group(1).upper()
            fields[current] = m.group(2).strip()
        elif current:
            fields[current] += "\n" + line
    if "TRANSLATION" not in fields:
        return json.dumps({"legibility": "partial", "printed_page_number": "",
                           "translation": text.strip(),
                           "notes": "model ignored the output format"})
    leg = fields.get("LEGIBILITY", "").lower().strip(" .*")
    leg = leg if leg in ("clear", "partial", "illegible") else "partial"
    return json.dumps({"legibility": leg,
                       "printed_page_number": fields.get("PRINTED_PAGE", "").strip(),
                       "translation": fields["TRANSLATION"].strip(),
                       "notes": fields.get("NOTES", "").strip()})


class MlxProvider:
    """Runs an open vision model locally on Apple Silicon via mlx-vlm (no network after the
    model is downloaded). `model` is a Hugging Face repo id or a local folder."""

    def __init__(self, model: str, max_tokens: int = 2048):
        from mlx_vlm import load
        from mlx_vlm.utils import load_config

        self.model_name, self.max_tokens = model, max_tokens
        self.model, self.processor = load(model)
        self.config = load_config(model)

    def translate(self, image: bytes, mime: str, temperature: float = 0.0, note: str = "") -> str:
        import tempfile

        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        prompt = apply_chat_template(self.processor, self.config,
                                     SYSTEM_PROMPT + ("\n" + note if note else "") + "\n" + TAGGED_FORMAT,
                                     num_images=1)
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            tmp.write(image)
            tmp.flush()
            out = generate(self.model, self.processor, prompt, [tmp.name], verbose=False,
                           max_tokens=self.max_tokens, temperature=temperature)
        return tagged_to_json(out.text if hasattr(out, "text") else str(out))


# ----------------------------------------------------------------- page handling
def page_files(folder: Path) -> list[Path]:
    """sheet001-1.png, sheet001-2.png, sheet002-1.png ... in reading order."""
    return sorted(folder.glob("sheet*.png"))


def default_label(path: Path) -> str:
    m = re.match(r"sheet0*(\d+)-(\d+)", path.stem)
    return f"{m.group(1)} (part {m.group(2)})" if m else path.stem


def load_labels(files: list[Path], labels_path: Path | None) -> list[str]:
    """Labels from translate_urdu_pdf.py's <pdf>.ocr.labels.json when it matches the
    page images one-to-one (e.g. '5 (right page)'), else labels derived from file names."""
    if labels_path and labels_path.exists():
        labels = json.loads(labels_path.read_text(encoding="utf-8"))
        if len(labels) == len(files):
            return labels
        print(f"WARNING: {labels_path.name} has {len(labels)} labels for {len(files)} images; "
              "using file names instead", file=sys.stderr)
    return [default_label(f) for f in files]


def image_bytes(path: Path, max_side: int) -> tuple[bytes, str]:
    from PIL import Image

    img = Image.open(path)
    if max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.convert("L").save(buf, format="PNG", optimize=True)
    return buf.getvalue(), "image/png"


# ----------------------------------------------------------------- result handling
def tile_cuts(img, n: int) -> list[int]:
    """Row positions (including 0 and the height) that cut the page into n strips. Each cut
    is moved to the emptiest nearby row (a gap between text lines) so no line is split."""
    g = img.convert("L")
    w, h = g.size
    if n <= 1 or h < n * 40:
        return [0, h]
    ink = [v / 255 for v in g.point(lambda p: 0 if p > 128 else 255).resize((1, h)).getdata()]
    smooth = [sum(ink[max(0, i - 2): i + 3]) / len(ink[max(0, i - 2): i + 3]) for i in range(h)]
    cuts = [0]
    for k in range(1, n):
        target, win = round(k * h / n), max(5, h // (4 * n))
        lo, hi = max(cuts[-1] + 20, target - win), min(h - 20, target + win)
        cuts.append(min(range(lo, hi + 1), key=lambda r: (smooth[r], abs(r - target))) if lo < hi else target)
    return cuts + [h]


def tiles_of(path: Path, n: int, max_side: int) -> list[tuple[bytes, str]]:
    """The page as n horizontal strips (or the whole page for n=1), PNG-encoded."""
    from PIL import Image

    img = Image.open(path)
    if n <= 1:
        return [image_bytes(path, max_side)]
    cuts = tile_cuts(img, n)
    out = []
    for a, b in zip(cuts, cuts[1:]):
        strip = img.crop((0, a, img.width, b)).convert("L")
        if max(strip.size) > max_side:
            scale = max_side / max(strip.size)
            strip = strip.resize((round(strip.width * scale), round(strip.height * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        strip.save(buf, format="PNG", optimize=True)
        out.append((buf.getvalue(), "image/png"))
    return out


def merge_tiles(results: list[dict]) -> dict:
    """Combine per-strip results into one page result. A strip that stayed stuck is replaced
    by [translation unreliable] so the rest of the page is still kept."""
    legs = [r["legibility"] for r in results]
    texts = [UNRELIABLE if r.get("flag") == "stuck" else r["translation"] for r in results]
    bad = [r.get("flag") == "stuck" or r["legibility"] == "illegible" for r in results]
    if all(l == "illegible" for l in legs):
        leg = "illegible"
    elif all(l == "clear" for l in legs) and not any(bad):
        leg = "clear"
    else:
        leg = "partial"
    page = next((r["printed_page_number"] for r in results if r["printed_page_number"]), "")
    notes = "; ".join(dict.fromkeys(r["notes"] for r in results if r["notes"]))
    merged = {"legibility": leg, "printed_page_number": page,
              "translation": "\n\n".join(t for t in texts if t), "notes": notes}
    if all(r.get("flag") == "stuck" for r in results):
        merged["flag"] = "stuck"
    return merged


def parse_result(text: str) -> dict:
    """Validate the model's JSON. Raises ValueError if it isn't the expected shape."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"not valid JSON: {exc}") from exc
    if not isinstance(data, dict) or data.get("legibility") not in ("clear", "partial", "illegible"):
        raise ValueError("missing or invalid 'legibility'")
    return {
        "legibility": data["legibility"],
        "printed_page_number": str(data.get("printed_page_number") or ""),
        "translation": str(data.get("translation") or "").strip(),
        "notes": str(data.get("notes") or "").strip(),
    }


MAX_STRIP_CHARS = 5000   # a page strip never holds this much text; a longer answer ran to the token cap


def looks_stuck(text: str) -> bool:
    """Repetition check for the model's translation. Tokens with no letters or digits (the
    ditto marks and rules in a catalog table) are ignored for the word checks so tables
    don't trip it, but long runs of them (a model looping on '...') still count."""
    if len(text) > MAX_STRIP_CHARS:
        return True
    if re.search(r"[.\-_=*~]{40,}", text):                    # a long run of dots/dashes
        return True
    marks = [w for w in text.split() if not re.search(r"\w", w)]
    run = best = 0
    for a, b in zip(marks, marks[1:]):
        run = run + 1 if a == b else 0
        best = max(best, run)
    if best >= 20:                                              # '... ... ...' 20+ times
        return True
    starts = Counter(" ".join(re.findall(r"\w+", s.lower())[:5])
                     for s in re.split(r"(?<=[.!?])\s+|\n+", text) if len(re.findall(r"\w+", s)) >= 5)
    if starts and starts.most_common(1)[0][1] >= 5:            # same sentence opening, reworded
        return True
    words = [w for w in text.split() if re.search(r"\w", w)]
    return is_degenerate(" ".join(words))


def translate_page(provider, path: Path, max_side: int, retries: int = 4,
                   sleep=time.sleep, tiles: int = 1) -> dict:
    """One page -> validated result dict. With tiles > 1 the page is cut into that many
    horizontal strips, each translated on its own, then merged in order."""
    parts = tiles_of(path, tiles, max_side)
    if len(parts) == 1:
        return translate_image(provider, path.name, *parts[0], retries=retries, sleep=sleep)
    results = []
    for i, (img, mime) in enumerate(parts, start=1):
        results.append(translate_image(provider, f"{path.name} strip {i}/{len(parts)}", img, mime,
                                       retries=retries, sleep=sleep, note=TILE_NOTE))
    return merge_tiles(results)


def translate_image(provider, name: str, img: bytes, mime: str, retries: int = 4,
                    sleep=time.sleep, note: str = "") -> dict:
    """One image -> validated result dict. Retries API errors with backoff, and retries a
    repeating/stuck translation once at a higher temperature before flagging it."""
    temperature, stuck_retried, attempt = 0.0, False, 0
    kwargs = {"note": note} if note else {}
    while True:
        try:
            result = parse_result(provider.translate(img, mime, temperature, **kwargs))
        except Exception as exc:  # API error, rate limit, or malformed JSON
            attempt += 1
            if attempt > retries:
                raise
            wait = min(60, 2 ** attempt)
            print(f"  {name}: {type(exc).__name__}: {str(exc)[:120]} "
                  f"(retry {attempt}/{retries} in {wait}s)", file=sys.stderr)
            sleep(wait)
            continue
        if result["translation"] and looks_stuck(result["translation"]):
            if not stuck_retried:
                stuck_retried, temperature = True, 0.4
                print(f"  {name}: repeating output, retrying once", file=sys.stderr)
                continue
            result["flag"] = "stuck"
        return result


def render_page(label: str, result: dict | None, error: str | None = None) -> str:
    head = f"## Page {label}"
    if result and result["printed_page_number"]:
        head += f" (printed page {result['printed_page_number']})"
    if error or result is None:
        return f"{head}\n\n*[not translated: {error or 'no result'}]*"
    body = result["translation"]
    if result.get("flag") == "stuck":
        body = f"*{UNRELIABLE}*"
    elif result["legibility"] == "illegible" or not body:
        body = f"*{ILLEGIBLE}*" if result["legibility"] == "illegible" else "*[no text on this page]*"
    parts = [head, "", body]
    if result["legibility"] == "partial":
        parts += ["", "*Partly legible.*"]
    if result["notes"]:
        parts += ["", f"> {result['notes']}"]
    return "\n".join(parts)


def load_cache(path: Path) -> dict[str, dict]:
    cache: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                cache[rec["file"]] = rec
    return cache


def run(files: list[Path], labels: list[str], provider, out: Path, *, max_side: int,
        delay: float, source_name: str, sleep=time.sleep, tiles: int = 1) -> list[str]:
    """Translate each page (using/updating the jsonl cache), write the Markdown, and
    return the names of pages that failed."""
    cache_path = out.with_name(out.name + ".jsonl")
    cache = load_cache(cache_path)
    failed: list[str] = []
    with cache_path.open("a", encoding="utf-8") as log:
        for n, (f, label) in enumerate(zip(files, labels), start=1):
            if f.name in cache and "error" not in cache[f.name]:
                continue
            try:
                rec = {"file": f.name, "label": label, "result": translate_page(provider, f, max_side, tiles=tiles)}
            except Exception as exc:
                rec = {"file": f.name, "label": label, "error": f"{type(exc).__name__}: {str(exc)[:200]}"}
                failed.append(f.name)
            cache[f.name] = rec
            log.write(json.dumps(rec, ensure_ascii=False) + "\n")
            log.flush()
            leg = rec.get("result", {}).get("legibility", "ERROR")
            print(f"[{n}/{len(files)}] page {label}: {leg}")
            sleep(delay)

    sections = [render_page(label, cache.get(f.name, {}).get("result"), cache.get(f.name, {}).get("error"))
                for f, label in zip(files, labels)]
    out.write_text(f"# Translation of {source_name}\n\n" + "\n\n---\n\n".join(sections) + "\n",
                   encoding="utf-8")
    return failed


# ----------------------------------------------------------------- CLI
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Translate page images to English with a vision LLM (Gemini).")
    ap.add_argument("pages", type=Path, nargs="?", help="Folder of sheetNNN-N.png page images (<pdf>.pages/)")
    ap.add_argument("-o", "--output", type=Path, help="Output .md (default: <pages folder>.vision.en.md)")
    ap.add_argument("--provider", choices=["gemini", "mlx"], default="gemini",
                    help="gemini = API (needs GEMINI_API_KEY); mlx = open model run locally with mlx-vlm")
    ap.add_argument("--model", default=None,
                    help="gemini: model name (or set GEMINI_MODEL); mlx: Hugging Face repo id or local folder")
    ap.add_argument("--max-tokens", type=int, default=2048, help="mlx only: cap on generated tokens per page")
    ap.add_argument("--list-models", action="store_true", help="List models your API key can use, then exit")
    ap.add_argument("--labels", type=Path, help="<pdf>.ocr.labels.json (default: found next to the pages folder)")
    ap.add_argument("--limit", type=int, help="Only the first N page images (for a cheap trial)")
    ap.add_argument("--skip", type=int, default=0, help="Skip the first N page images")
    ap.add_argument("--max-side", type=int, default=2200, help="Downscale images whose longest side exceeds this")
    ap.add_argument("--delay", type=float, default=0.5, help="Seconds to wait between requests")
    ap.add_argument("--tiles", type=int, default=1,
                    help="Cut each page into this many horizontal strips (cut between text lines) and "
                         "translate each separately; helps models that loop on dense pages")
    ap.add_argument("--dry-run", action="store_true", help="List what would be sent; make no API calls")
    return ap.parse_args(argv)


def get_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        sys.exit("Set GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment.")
    return key


def main(argv=None):
    a = parse_args(argv)
    if a.list_models:
        print("\n".join(GeminiProvider.list_models(get_api_key())))
        return
    if not a.pages or not a.pages.is_dir():
        sys.exit("Give the folder of page images (the <pdf>.pages folder).")
    files = page_files(a.pages)
    if not files:
        sys.exit(f"No sheet*.png images in {a.pages}")
    labels_path = a.labels
    if not labels_path and a.pages.name.endswith(".pages"):
        labels_path = a.pages.with_name(a.pages.name[: -len(".pages")] + ".ocr.labels.json")
    labels = load_labels(files, labels_path)
    files, labels = files[a.skip:], labels[a.skip:]
    if a.limit:
        files, labels = files[:a.limit], labels[:a.limit]
    if not files:
        sys.exit("Nothing to translate after --skip/--limit.")
    out = a.output or a.pages.with_name(a.pages.name.removesuffix(".pages") + ".vision.en.md")
    if a.dry_run:
        print(f"Would translate {len(files)} page images from {a.pages} -> {out}")
        return
    if a.provider == "mlx":
        if not a.model:
            sys.exit("Give --model (a Hugging Face repo id or local folder) for --provider mlx.")
        provider = MlxProvider(a.model, a.max_tokens)
    else:
        model = a.model or os.environ.get("GEMINI_MODEL")
        if not model:
            sys.exit("Give --model (or set GEMINI_MODEL); run --list-models to see what your key can use.")
        provider = GeminiProvider(model, get_api_key())
    print(f"{len(files)} pages with {a.provider}:{a.model or model} -> {out}")
    failed = run(files, labels, provider, out, max_side=a.max_side, delay=a.delay, tiles=a.tiles,
                 source_name=a.pages.name.removesuffix(".pages"))
    print(f"Wrote {out}" + (f"; {len(failed)} pages failed: {', '.join(failed)}" if failed else ""))
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
