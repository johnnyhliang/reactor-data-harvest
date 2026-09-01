# reactor-data-harvest

A feasibility prototype for the **data-scarcity problem** in the NuclearDiffusion
project (arXiv:2608.04030). The paper could only assemble ~1,000 captioned nuclear
images and named "increase the dataset size to hundreds of thousands" as its top
future-work item. This prototype demonstrates a **real, license-clean, already-
captioned source the paper never used**: DOE technical reports on OSTI.gov.

## The idea

The paper mined Elsevier journals, 3 handbooks, and news sites. From 11,600
journal PDFs it kept only **119** images (its own filter discarded 41,342 → 119).

OSTI.gov is different and better for this purpose:
- **Public domain.** DOE-contract reports are U.S. government works — no license risk.
- **Captions come free.** Every figure has an author-written caption we pair to it.
- **High yield.** In this prototype: **~11 captioned figures per PDF**, vs. the
  paper's ~0.01 usable images per journal PDF.
- **On-topic.** Reports are *about* reactor cores, fuel assemblies, casks, etc.

## What's here

| File | Role | Status |
|---|---|---|
| `harvest_osti.py` | Search OSTI → download full-text PDFs → extract (figure, caption) pairs → HuggingFace imagefolder format | **Working** (986 figures, 12/PDF) |
| `harvest_patents.py` | Same for US patents (Google Patents, CPC G21C nuclear); parses per-figure descriptions | **Working** (15.7 figs/patent; Google rate-limits rapid runs — has backoff) |
| `recaption.py` | VLM stage: drop plots/charts, enrich terse captions. Backends: local **ollama** (free), anthropic, or passthrough | **Working**; needs one real run on a GPU/Ryzen box |
| `PIPELINE.md` | Plain-English explanation of how the harvesting works | doc |
| `WRITEUP.md` | Lab-facing writeup with results + caveats | doc |
| `output_big/`, `output_patents/` | `images/` + `metadata.jsonl` — the paper's fine-tuning format | generated |

New to this? Read **`PIPELINE.md`** first — it explains, from scratch, how the code
turns a search query into captioned images (and confirms nothing is AI-generated).

## Run it

```bash
pip install requests pymupdf pillow
python3 harvest_osti.py --queries "PWR fuel assembly" "reactor core cross section" --rows 8
# then (optional) filter + enrich:
export ANTHROPIC_API_KEY=...        # optional; without it, passthrough mode
python3 recaption.py --in output/metadata.jsonl --out output/metadata_filtered.jsonl
```

Load downstream exactly like the paper's data:
```python
from datasets import load_dataset
ds = load_dataset("imagefolder", data_dir="output")
```

## Prototype results (2 queries, 14 PDFs)

- 14/14 PDFs had downloadable full text
- **155 captioned figures** extracted (~11/PDF)
- Manual spot-check: clean labeled reactor-core cross-sections, spent-fuel
  facility layouts, cutaways — the exact "nuclear schematic" class the paper wants
- Mixed in: data plots (temperature/burnup curves) — these are what `recaption.py`
  is meant to filter out downstream

## Honest limitations

- **Raw harvest is unfiltered.** It pulls plots/charts too; the schematic-vs-plot
  filter (`recaption.py`) is a stub until wired to a vision model.
- **Caption pairing is heuristic** (nearest image to each "Figure N" block). Works
  well on single-figure pages; multi-panel pages can mis-pair. Spot-check before use.
- **In-text "Figure N ..." references** are filtered by a verb heuristic; a few slip through.
- **Dedup across reports** not yet done (same figure can appear in related reports).
- Scope is OSTI only. Patents (Google Patents/USPTO) and NRC ADAMS are the obvious
  next sources — same pattern, also public domain, even richer captions.

## Why this matters for the lab

This is the "reliable source of captioned images" alternative to generating
synthetic variations with Gemini. It's real data, legally clean, and scales:
a few hundred targeted queries × ~11 figures/PDF plausibly reaches the thousands
of images the paper says it needs — before any augmentation.
