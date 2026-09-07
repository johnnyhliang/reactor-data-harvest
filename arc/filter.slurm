#!/usr/bin/env bash
#SBATCH --job-name=nuc-recaption
#SBATCH --partition=gpu               # Great Lakes GPU partition (spgpu also works)
#SBATCH --gpus=1                      # 1 GPU is plenty for a 7B/32B VLM
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=arc/logs/recaption-%j.out
### EDIT THIS: your Slurm account (find it with `my_accounts` on the login node),
### or pass on the command line:  sbatch --account=<acct> arc/filter.slurm
#SBATCH --account=CHANGE_ME

# filter.slurm -- Stage 2: VLM filter + re-caption on a GPU node (NO internet needed).
# The Ollama model must already be pulled on the login node (see RUN_ON_ARC.md),
# because ~/.ollama is on shared home storage and this node cannot reach the net.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p arc/logs

MODEL="${MODEL:-qwen2.5vl:32b}"       # override: MODEL=qwen2.5vl:7b sbatch ...
PY=.venv/bin/python

# --- start a private Ollama server on this node ---------------------------
export OLLAMA_HOST="127.0.0.1:11434"
export OLLAMA_MODELS="$HOME/.ollama/models"   # models pulled on the login node
ollama serve > "arc/logs/ollama-${SLURM_JOB_ID}.log" 2>&1 &
OLLAMA_PID=$!
trap 'kill $OLLAMA_PID 2>/dev/null || true' EXIT

# wait for it to come up
for i in $(seq 1 30); do
  curl -sf "http://$OLLAMA_HOST/api/tags" >/dev/null && break
  sleep 2
done

# --- filter both harvests -------------------------------------------------
for d in output_big output_patents; do
  [ -f "$d/metadata.jsonl" ] || { echo "skip $d (no metadata)"; continue; }
  echo "== filtering $d with $MODEL =="
  $PY recaption.py --backend ollama --model "$MODEL" \
      --ollama-host "http://$OLLAMA_HOST" \
      --in "$d/metadata.jsonl" --out "$d/metadata_filtered.jsonl"
done

echo
echo "== filtered dataset ready =="
wc -l output_*/metadata_filtered.jsonl 2>/dev/null || true
