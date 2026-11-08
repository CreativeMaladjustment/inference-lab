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
4. **Output** – paragraphs are rebuilt in order; pages are separated by `---`.

### Requirements
- macOS on Apple Silicon (runs on CPU elsewhere, slowly), Python 3.10–3.12
- Poppler: `brew install poppler`
- Disk/RAM for the NLLB 1.3B model (~5 GB download, ~5 GB RAM in fp32)

### Setup
```bash
cd urdu_translate
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip check
```

> **Dependency note:** `surya-ocr` pins a narrow `transformers` range; NLLB works with most 4.x, so `transformers` is pinned to Surya's requirement. Don't `pip install -U transformers`. The pins in `requirements.txt` were not test-installed by the author; if `pip check` complains, align with `pip show surya-ocr`. The CI job *requirements.txt resolves* reports whether they currently resolve.
>
> Current Surya has no `ur` language argument; it is multilingual and handles Urdu itself.

### Run
```bash
python translate_urdu_pdf.py /path/to/book.pdf -o book.en.md --save-ocr book.ocr.json
```
Try `--first-page 1 --last-page 3` first.

| Option | Default | Purpose |
|---|---|---|
| `-o, --output` | `<pdf>.en.md` | Output Markdown path |
| `--dpi` | 200 | Page render resolution |
| `--first-page / --last-page` | all | 1-based page range |
| `--ocr-batch` | 2 | Pages per OCR batch (memory) |
| `--chunk-size` | 400 | Max characters per chunk |
| `--batch-size` | 4 | Chunks per translation batch |
| `--beams` | 4 | Beam-search width |
| `--max-input-tokens` | 256 | Truncation limit per chunk |
| `--fp16` | off | Half-precision model (less RAM, may be less stable on MPS) |
| `--save-ocr FILE` | – | Cache OCR JSON; reused on re-run (OCR is the slow step) |

The script sets `TORCH_DEVICE=mps` and `PYTORCH_ENABLE_MPS_FALLBACK=1` before importing Surya, and places NLLB on `mps`; it falls back to CPU with a warning if MPS is unavailable.

### Tests & CI
Tests stub out OCR and the NLLB model, so they need no model downloads:
```bash
pip install torch && pip install -r requirements-test.txt
python -m pytest
```
They cover device selection, paragraph grouping, chunking, translation ordering / forced English BOS, Markdown assembly, and the CLI end to end (including OCR caching).

GitHub Actions ([`urdu-translate.yml`](.github/workflows/urdu-translate.yml)) runs them on Ubuntu and macOS (Apple Silicon), Python 3.10 and 3.12, plus a non-blocking check that `requirements.txt` resolves. CI does **not** run real Surya/NLLB inference, and hosted macOS runners have no usable MPS, so real-model and MPS behaviour must be verified locally.
