# inference-lab
A local testbench for evaluating open-weights models, orchestration pipelines, and AI toolchains.

## Projects

| Project | What it does |
|---|---|
| [`urdu_translate/`](urdu_translate/) | Translate large multi-page Urdu PDFs to English Markdown (Surya OCR + NLLB-200, Apple Silicon `mps`) |

---

## urdu_translate

Pipeline: **PDF → page images → OCR → chunking → translation → Markdown**

1. **OCR** – `pdf2image` renders pages; [Surya](https://github.com/datalab-to/surya) extracts text lines, which are regrouped into paragraphs by vertical gap.
2. **Chunking** – LangChain's `RecursiveCharacterTextSplitter` splits each paragraph on natural boundaries (newlines, Urdu `۔` / `؟`, then other punctuation, then words). Default 400 chars/chunk, no overlap.
3. **Translation** – `facebook/nllb-200-distilled-1.3B`, `urd_Arab` → `eng_Latn`, batched, with a `tqdm` progress bar.
4. **Output** – paragraphs are rebuilt in order, under a `## Page N of <file>` heading per page, separated by `---`.

### Requirements
- macOS on Apple Silicon (runs on CPU elsewhere, slowly), Python 3.13 (3.14 isn't usable yet: Surya 0.17 needs Pillow<11, which has no 3.14 wheels)
- Poppler and Pango: `brew install poppler pango` (Pango is only needed for `--to-pdf`)
- Disk/RAM for the NLLB 1.3B model (~5 GB download, ~5 GB RAM in fp32)

### Setup
```bash
cd urdu_translate
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip check
```

> **Dependency note:** `surya-ocr` 0.17 requires `transformers>=4.56.1` and `Pillow<11`; we pin `transformers==4.56.1`. Don't upgrade `surya-ocr` past 0.17.x: Surya v2 (0.20+) needs transformers 5 and a vllm/llama.cpp server and has a different API than this script. The pins resolve together on Python 3.13 (Linux dry run); they haven't been installed or run on a Mac.
>
> Current Surya has no `ur` language argument; it is multilingual and handles Urdu itself.

### Run
Process one PDF, or every `*.pdf` in a folder (the 1.3B model is loaded once for the whole batch):
```bash
python translate_urdu_pdf.py ~/temp/pdfs-to-process --output-dir ~/temp/pdfs-translated --save-ocr
```
Try a single file with `--first-page 1 --last-page 3` first.

**Page count:** at the start of each PDF the log states `PDF has N pages` (and the range, if you used `--first-page/--last-page`), and warns if the number of pages rendered differs.

**Progress:** everything is written to stdout, including the `tqdm` bars (OCR per page, translation per chunk) and one log line per OCR page, e.g. `OCR page 12: 41 lines, 7 paragraphs, 1234 chars` (with a `(no text detected)` flag for blank pages). Page numbers respect `--first-page`. Only model-download bars and library warnings still go to stderr.

**Log file:** every run writes `urdu_translate-<timestamp>.log` (in `--output-dir`, or next to the PDF/folder; override with `--log-file`). It mirrors stdout **and** stderr, so it includes library warnings and tracebacks, with progress bars collapsed to their final state. The header records the command, Python/OS, device (MPS availability) and the versions of torch, transformers, surya-ocr etc., and native crashes (e.g. an MPS segfault) get a `faulthandler` traceback. A failure in one PDF is logged with its traceback and the batch carries on; the run ends with a summary and exits non-zero if anything failed. To troubleshoot, send the `.log` file.

**Output naming:** `book.pdf` → `book.en.md` (and `book.ocr.json` with `--save-ocr`), in `--output-dir`, or next to the PDF if that's not given. Each page in the Markdown starts with a visible `## Page N of book.pdf` heading, and pages are separated by `---`. Provenance is page-level only (no bounding boxes or per-paragraph references).

| Option | Default | Purpose |
|---|---|---|
| `-o, --output` | `<pdf name>.en.md` | Output path (single PDF only) |
| `--output-dir` | next to each PDF | Folder for outputs |
| `--dpi` | 200 | Page render resolution |
| `--first-page / --last-page` | all | 1-based page range |
| `--ocr-batch` | 2 | Pages per OCR batch (memory) |
| `--chunk-size` | 400 | Max characters per chunk |
| `--batch-size` | 4 | Chunks per translation batch |
| `--beams` | 4 | Beam-search width |
| `--max-input-tokens` | 256 | Truncation limit per chunk |
| `--fp16` | off | Half-precision model (less RAM, may be less stable on MPS) |
| `--to-pdf` | off | Also render `<pdf name>.en.pdf` from the Markdown (each source page starts a new PDF page) |
| `--save-ocr` | off | Cache OCR as `<pdf name>.ocr.json`; reused on re-run (OCR is the slow step) |

The script sets `TORCH_DEVICE=mps` and `PYTORCH_ENABLE_MPS_FALLBACK=1` before importing Surya, and places NLLB on `mps`; it falls back to CPU with a warning if MPS is unavailable.

### Word frequencies & concordance
`word_stats.py` counts words in the Urdu OCR text (`*.ocr.json`, written by `--save-ocr`), standard library only, runs in seconds:
```bash
python word_stats.py ~/temp/pdfs-to-process
python word_stats.py ~/temp/pdfs-to-process --search سائنس --context 6    # look a word up
```
Writes to `<folder>/word_stats/`: `word_frequencies.csv` (rank, word, total, a count column per book, pages used on), `concordance.csv` (every occurrence with book, PDF page, paragraph and surrounding words, target in `**bold**`), and `summary.txt`. CSVs have a BOM so Excel/Numbers open Urdu correctly. Spelling variants are unified before counting (Arabic vs Urdu yeh/kaf/heh, vowel marks, tatweel, zero-width joiners); numerals and punctuation are ignored (`--keep-numbers` to count numerals). `--translate` adds an `english_gloss_machine` column (and `= gloss` in the summary) for the `--translate-top N` (default 300) most frequent words, using the same NLLB model; glosses are cached in `word_translations.json`, which you can hand-edit and re-run. These are single-word machine translations without context, so treat them as hints: they are unreliable for Arabic words, ambiguous words and OCR garbage. Counts are of OCR text, so OCR misreadings show up as rare spurious words. Page numbers are PDF pages, matching the `## Page N` headings in the `.md`.

### Tests & CI
Tests stub out OCR and the NLLB model, so they need no model downloads:
```bash
pip install torch && pip install -r requirements-test.txt
python -m pytest
```
They cover device selection, paragraph grouping, chunking, translation ordering / forced English BOS, Markdown assembly, and the CLI end to end (including OCR caching, folder input and output naming).

GitHub Actions ([`urdu-translate.yml`](.github/workflows/urdu-translate.yml)) runs them on Ubuntu with Python 3.13, plus a non-blocking check that `requirements.txt` resolves. CI does **not** run real Surya/NLLB inference or exercise MPS, so real-model and MPS behaviour must be verified locally.
