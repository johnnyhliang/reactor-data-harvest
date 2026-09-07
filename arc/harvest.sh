#!/usr/bin/env bash
# harvest.sh -- Stage 1: run on an ARC LOGIN NODE (needs internet, no GPU).
# Downloads PDFs from OSTI + Google Patents and extracts (figure, caption) pairs.
# GPU compute nodes on Great Lakes usually have NO outbound internet, so this
# MUST run on the login node (or a data-transfer node), not inside the Slurm job.
set -euo pipefail
cd "$(dirname "$0")/.."          # repo root

# --- python env (first run only creates it) -------------------------------
if [ ! -d .venv ]; then
  module load python 2>/dev/null || true    # Lmod; harmless if already present
  python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q -r requirements.txt
fi
PY=.venv/bin/python

# --- OSTI: broad query sweep (~12 captioned figs/PDF) ----------------------
$PY harvest_osti.py --rows 15 --out output_big --queries \
  "PWR fuel assembly" "BWR fuel assembly" "reactor core cross section" \
  "control rod assembly" "control rod drive mechanism" "spent fuel storage cask" \
  "dry cask storage" "TRISO fuel particle" "reactor pressure vessel internals" \
  "steam generator nuclear" "fuel rod cladding" "reactor coolant system schematic" \
  "molten salt reactor design" "sodium fast reactor" "high temperature gas reactor" \
  "SMR reactor module" "microreactor design" "fuel pellet stack" \
  "reactor containment structure" "pressurizer nuclear" "core barrel baffle" \
  "grid spacer fuel assembly" "burnable absorber rod" "reactor vessel head" \
  "heat exchanger primary loop" "reactor cavity layout" "neutron reflector assembly" \
  "spent fuel pool rack" "fuel handling machine" "reactor internals cutaway"

# --- US patents (Google Patents throttles; keep --pages small) -------------
# If you hit repeated 503s, wait a few minutes and re-run -- it's resumable.
$PY harvest_patents.py --pages 2 --out output_patents --queries \
  "fuel assembly" "control rod drive" "reactor pressure vessel" \
  "nuclear fuel rod" "spent fuel cask" "reactor core structure" \
  "steam generator" "fuel spacer grid" "reactor coolant pump"

echo
echo "== raw harvest done =="
wc -l output_big/metadata.jsonl output_patents/metadata.jsonl 2>/dev/null || true
echo "Next: sbatch arc/filter.slurm   (runs the VLM filter on a GPU node)"
