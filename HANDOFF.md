# Handoff / resume guide

Everything needed to pick this up on another machine (e.g. the Ryzen AI box).
The scripts fully regenerate all data — no local files are precious.

## 0. Setup on a new machine

```bash
git clone https://github.com/johnnyhliang/reactor-data-harvest.git
cd reactor-data-harvest
pip install -r requirements.txt        # requests, pymupdf, pillow
```

## 1. Regenerate the harvested data

The `output_*/` folders are NOT in git (regenerable). Recreate them:

```bash
# DOE reports (OSTI) — ~12 captioned figures/PDF
python3 harvest_osti.py --queries "PWR fuel assembly" "BWR fuel assembly" \
  "reactor core cross section" "control rod assembly" "spent fuel storage cask" \
  "TRISO fuel particle" "reactor pressure vessel internals" "steam generator nuclear" \
  "fuel rod cladding" "reactor coolant system schematic" --rows 12 --out output_big

# US patents — ~15 figures/patent (Google rate-limits; keep --pages small, has backoff)
python3 harvest_patents.py --queries "fuel assembly" "control rod drive" \
  "reactor pressure vessel" --pages 1 --out output_patents
```

Note: OSTI's search index changes over time, so re-runs return a functionally
equivalent (not bit-identical) set of figures. That's expected.

## 2. Filter + enrich (the key pending step)

Turns raw figures (incl. plots) into a clean, richly-captioned dataset. Runs
locally and free on the Ryzen box.

```bash
ollama pull qwen2.5vl:32b               # verify tag with `ollama list`
python3 recaption.py --backend ollama --model qwen2.5vl:32b \
    --in output_big/metadata.jsonl --out output_big/metadata_filtered.jsonl
```

- Model choice: `qwen2.5vl:32b` (best quality/speed on 128GB; reads diagrams+text
  better than `llama3.2-vision:11b`). Use `qwen2.5vl:7b` for fast iteration.
- Resumable — safe to stop/restart. Prints a kept/dropped count + category
  breakdown. **That kept-count is the real usable-image number for the writeup**
  (currently only an estimate: ~20–30% of raw).

## 3. Remaining todos

- [ ] Run step 2 → record the real usable-image count; update `WRITEUP.md` §4.
- [ ] Scale OSTI harvest to a few hundred queries for a larger dataset.
- [ ] Resume patent harvest with request pacing (avoid Google 503s).
- [ ] Deduplicate figures across documents (not yet implemented).
- [ ] (optional) `combine_datasets.py` to merge OSTI + patent outputs into one folder.
- [ ] Decide repo home — currently under `johnnyhliang`; consider transfer to the
      lab org `aims-umich`.
- [ ] Hand the ablation experiment (baseline vs +harvest vs +recaptioned; measure
      KID/CMMD + failure count) to whoever owns fine-tuning + GPUs.

## 4. Where things are

| What | Where |
|---|---|
| How the code works (plain English) | `PIPELINE.md` |
| Lab-facing writeup + caveats | `WRITEUP.md` |
| Usage / quick start | `README.md` |
| Harvesters + filter stage | `harvest_osti.py`, `harvest_patents.py`, `recaption.py` |
| Sample outputs (10 figures) | `samples/` |

## 5. Key caveats (don't lose these)

- Raw yield is ~20–30% usable (rest are plots/charts/tangential site photos).
- Data is diagram/schematic-skewed → complements, does not replace, real photos.
- Caption pairing is heuristic; spot-check multi-panel pages.
- All sources are public domain (OSTI = US-gov works; US patents); source IDs are
  stored for provenance.
