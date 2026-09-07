# Running the full pipeline on ARC (Great Lakes)

Goal: produce the complete filtered, richly-captioned dataset
(`output_big/metadata_filtered.jsonl` + `output_patents/metadata_filtered.jsonl`)
in HuggingFace imagefolder format, ready for the paper's fine-tuning code.

## The one thing that shapes everything

On Great Lakes, **login nodes have internet; GPU compute nodes do not.**
So the pipeline runs in two places:

| Stage | Where | Why |
|---|---|---|
| 1. Harvest (download PDFs, extract figures) | **login node** | needs internet, no GPU |
| 2. Filter + re-caption (VLM) | **Slurm GPU job** | needs GPU, no internet |

Do **not** try to run the harvest inside the GPU job — it will hang on downloads.

---

## Step 0 — clone + find your Slurm account (login node)

```bash
git clone https://github.com/johnnyhliang/reactor-data-harvest.git
cd reactor-data-harvest
my_accounts            # <- note your account name (e.g. mradaideh0 or a class acct)
```

## Step 1 — pre-pull the VLM (login node, needs internet)

The GPU node can't download the model, so pull it now into your shared home
(`~/.ollama`). Install Ollama to user space if it isn't already available:

```bash
# one-time user-space install (no root needed)
curl -fsSL https://ollama.com/install.sh | sh   # or: module load ollama (if ARC provides it)

ollama serve &                 # start it briefly just to pull
sleep 5
ollama pull qwen2.5vl:32b      # ~20 GB; use qwen2.5vl:7b if short on home quota / VRAM
ollama list                    # confirm the tag
kill %1                        # stop the temporary server
```

VRAM guide: `qwen2.5vl:32b` (~24 GB, best quality) fits an A40/A100; on a 16 GB
V100 use `qwen2.5vl:7b`.

## Step 2 — harvest (login node)

```bash
bash arc/harvest.sh
```

This creates `.venv`, installs deps, and writes `output_big/` and
`output_patents/` (raw figures + captions). ~30 OSTI queries × 15 reports plus a
patent sweep — expect a few thousand raw figures. Re-runnable; safe to Ctrl-C and
restart (patents are resumable if Google rate-limits you).

## Step 3 — filter + re-caption (GPU job)

```bash
sbatch --account=<your_acct> arc/filter.slurm
# faster/smaller model instead:
# MODEL=qwen2.5vl:7b sbatch --account=<your_acct> arc/filter.slurm
```

Watch it:

```bash
squeue --me
tail -f arc/logs/recaption-*.out
```

The job starts a private Ollama server on the node, then runs `recaption.py`
over both harvests. It drops plots/charts/tangential photos and rewrites terse
captions into rich ones. **Resumable** — if the job times out, just `sbatch`
again and it continues where it left off.

## Step 4 — the deliverable

When it finishes, the log prints the real kept/dropped counts. The clean dataset:

```
output_big/metadata_filtered.jsonl        # OSTI, kept rows only, enriched captions
output_patents/metadata_filtered.jsonl    # patents, same
```

The kept-count is **the real usable-image number** — update `WRITEUP.md` §4 with
it (replaces the ~20–30% estimate). Load it exactly like the paper does:

```python
from datasets import load_dataset
ds = load_dataset("imagefolder", data_dir="output_big")   # uses metadata.jsonl
```

To train on the *filtered* set, point the loader at a folder whose
`metadata.jsonl` is the filtered file (copy/rename `metadata_filtered.jsonl` →
`metadata.jsonl` in a fresh dir, or symlink), since `imagefolder` looks for that
exact name.

---

## Scaling further

- More OSTI images: add queries to `arc/harvest.sh` or raise `--rows`.
- More patents: raise `--pages` (but pace it — Google throttles; the harvester
  backs off and is resumable).
- Dedup across documents is still a TODO (`HANDOFF.md`) — worth doing before a
  large training run.

## Troubleshooting

- **Job dies immediately / `curl: connection refused`** — Ollama didn't start.
  Check `arc/logs/ollama-<jobid>.log`. Usually a bad/missing model pull (redo
  Step 1) or the GPU node genuinely has no `~/.ollama/models`.
- **`CHANGE_ME` account error** — pass `--account=<acct>` to `sbatch` or edit the
  `#SBATCH --account=` line in `arc/filter.slurm`.
- **Harvest hangs** — you're on a compute node; run `arc/harvest.sh` on the login
  node instead.
- **Patents all 503** — Google rate-limited your IP; wait ~10 min and re-run
  (resumable).
