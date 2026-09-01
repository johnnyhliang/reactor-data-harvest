# Expanding the NuclearDiffusion dataset with public-domain document mining

**Author:** Jon Liang
**Status:** feasibility prototype
**Context:** NuclearDiffusion (arXiv:2608.04030) — data-scarcity problem

---

## 1. Problem

The NuclearDiffusion study fine-tunes text-to-image models on nuclear imagery and
shows fine-tuned SDXL beats commercial models on technical prompts. Its binding
constraint is **data**: only ~1,000 captioned images could be assembled (348 from
handbooks, 119 from 11,600 journal PDFs, 533 from news sites). The paper names two
related limitations:

- **Quantity:** it calls for scaling "to hundreds of thousands" of images.
- **Caption quality:** source captions are "short and not sufficiently descriptive."

The journal route was especially inefficient: an automated pipeline extracted
41,342 candidate images from 11,600 PDFs and, after aggressive filtering, kept
only **119** (~0.01 usable images per PDF).

## 2. Idea

Mine **public-domain technical documents the paper did not use**, where figures
already carry author-written captions:

- **OSTI.gov** — the U.S. DOE technical-report repository. Reports authored under
  DOE contracts are U.S. government works (public domain), full-text PDFs are
  openly downloadable, and every figure has a caption.
- **US patents** (via Google Patents, class G21C = nuclear reactors) — public
  domain, with unusually rich figure descriptions (patent law requires each
  drawing be described in text).

Both are legally clean and large: OSTI returns 172,075 records for "nuclear
reactor"; Google Patents returns ~100,000 for nuclear fuel-assembly queries.

## 3. Method

A four-stage pipeline (details in `PIPELINE.md`):

1. **Search** each source's API for a nuclear query.
2. **Download** each document's full-text PDF.
3. **Extract + pair** — pull each figure image and attach its caption. For reports,
   pair by on-page position (caption nearest an image). For patents, parse the
   "Brief Description of the Drawings" section and match by figure number.
4. **Filter + enrich** (`recaption.py`) — a vision-language model drops data
   plots/charts and rewrites terse captions into descriptive ones. Runs locally
   (Ollama) at zero cost.

Output is HuggingFace imagefolder format (`images/` + `metadata.jsonl`), which the
paper's fine-tuning code consumes directly.

## 4. Results (prototype)

| Metric | OSTI | Patents |
|---|---|---|
| Queries run | 10 | 3 (before rate-limit) |
| Documents with full-text PDF | 82 / 82 | 20 / 20 |
| Raw captioned figures | 986 | 314 |
| Yield | **12.0 figures / PDF** | **15.7 figures / patent** |
| License | public domain (US gov) | public domain (US patents) |

For comparison, the paper's journal route yielded ~0.01 usable images/PDF. Even
after quality filtering (below), OSTI alone is ~2–3 usable images/document — a
**100–300× efficiency improvement** on the per-document basis.

**Spot-check (manual, honest):** of a random sample of raw OSTI figures, roughly:
- ~20% are on-target schematics/cross-sections/component photos (the goal),
- ~40% are data plots/charts (to be dropped),
- ~40% are tangential (site aerials, rail infrastructure, borderline).

So the realistic **usable fraction is ~20–30%** of the raw harvest. Applied to the
986 raw figures → an estimated ~200–300 clean images from ~82 documents. (The
exact number will come from the `recaption.py` filtering run.)

**Scaling estimate:** a few hundred targeted queries across both sources plausibly
reaches **~15,000–40,000 clean, captioned, public-domain images** — 15–40× the
paper's entire current dataset — before any augmentation.

## 5. How this should move the paper's metrics

Two independent levers, affecting different metrics (see the paper's Section 4):

- **More images (quantity)** → lowers KID / CMMD (better coverage of the real
  image distribution) and reduces the human "concept-failure" count (the 84/250/300
  fully-failed prompts).
- **Better captions (quality)** → improves text–image alignment (CLIP score,
  ImageReward) and reduces concept failures, because captions are the model's only
  conditioning signal.

**Proposed experiment (the measurable contribution):** an ablation —

| Training set | Hypothesized effect |
|---|---|
| 1,000 original (baseline) | reproduce paper |
| 1,000 + OSTI/patent harvest | KID/CMMD ↓, failures ↓ |
| same, VLM-recaptioned | CLIP/ImageReward ↑, failures ↓ further |

Report the actual movement (e.g. "CMMD X→Y, failures 84→Z"). That number is the
deliverable.

## 6. Caveats (read before scaling)

- **Diagram/schematic-skewed.** OSTI and patents are heavy on CAD/line-drawings,
  light on photorealistic images. This **complements** the paper's web-scraped
  photos; it does not replace them. Mixing ratios should be deliberate.
- **Raw yield is inflated.** Only ~20–30% of raw figures survive quality filtering;
  the headline "986 / 314" are raw counts, not usable counts.
- **Patent drawings are a narrow visual style** (black-and-white, reference
  numerals). Great captions, stylized images.
- **Caption pairing is heuristic.** Reliable on single-figure pages; multi-panel
  pages can mis-pair. Spot-check a sample before training.
- **Filtering not yet run at scale.** `recaption.py` is implemented and tested but
  the real VLM pass (on the Ryzen box) is pending — usable counts are estimates
  until then.
- **Rate limits.** Google Patents throttles rapid requests (503); the harvester
  now backs off, but large runs need pacing.
- **Provenance/attribution.** Public domain, but keep source IDs (we store
  `osti_id` / `patent_number`) for traceability.

## 7. Next steps

1. Run `recaption.py` on the Ryzen box (local Ollama VLM) → real usable-image count.
2. Scale the OSTI harvest to a few hundred queries.
3. Resume the patent harvest with request pacing.
4. Deduplicate across documents.
5. Hand the ablation experiment to whoever owns fine-tuning + GPUs.

## 8. Artifacts

- `harvest_osti.py`, `harvest_patents.py` — working harvesters
- `recaption.py` — filter + enrich stage (local-VLM ready)
- `PIPELINE.md` — how the code works
- `output_big/` — 986 raw OSTI figures (demo)
