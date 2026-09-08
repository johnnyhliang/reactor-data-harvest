# Running the pipeline on the Ryzen box (+ how to VPN/SSH into ARC)

The Ryzen AI box (128 GB unified memory) runs the whole pipeline start-to-finish
in one place: internet and GPU on the same machine, root access, no Slurm. This is
the recommended way to run it.

The ARC VPN/SSH section is included first only for the times you *do* want the
cluster (e.g. a large multi-GPU fine-tuning run). For just harvesting + recaptioning,
skip straight to [Part B](#part-b--run-the-pipeline-on-the-ryzen-box).

---

## Part A — VPN + SSH into ARC (U-M Great Lakes), from Linux

U-M moved its VPN to a **Cisco AnyConnect gateway behind Okta SSO**. Plain
`openconnect` can't complete the Okta browser flow ("No SSO handler"), so use the
`openconnect-sso` wrapper, which opens a real browser window for Okta.

### One-time setup (Fedora)

```bash
# build deps for the wrapper's dependencies
sudo dnf install -y libxml2-devel libxslt-devel python3-devel gcc python3.12

# install pipx, then openconnect-sso built against Python 3.12
# (3.13 can't compile the pinned lxml; 3.12 uses a prebuilt wheel)
pip install --user pipx
python3 -m pipx install --python /usr/bin/python3.12 openconnect-sso
python3 -m pipx inject --force openconnect-sso "setuptools<81"   # provides pkg_resources
python3 -m pipx ensurepath        # puts ~/.local/bin on PATH (open a new terminal after)
```

### Connect (each time)

```bash
openconnect-sso --server umvpn.umnet.umich.edu --authgroup "UMVPN - Only U-M Traffic"
```

1. A browser window opens → sign in with uniqname + UMICH password → approve **Okta Verify**.
2. It runs openconnect and prompts for your **local sudo password** to bring up the tunnel.
3. Wait for `Established DTLS connection` / `Connected as …`, then **leave that terminal open**.

Then, in a second terminal:

```bash
ssh <uniqname>@greatlakes.arc-ts.umich.edu     # UMICH password + Okta/Duo
```

Notes / gotchas learned the hard way:
- The old `vpn.umich.edu` hostname is dead — the live gateway is `umvpn.umnet.umich.edu`.
- Off campus you **must** be on the VPN; ARC's SSH port is firewalled otherwise.
- On ARC you are **never root** — `sudo` fails ("not in the sudoers file"). Install
  user-space tools into `$HOME` and call them by full path instead (see the ARC
  runbook `RUN_ON_ARC.md`).
- The ARC two-node split (login node has internet, GPU nodes don't) is why the
  cluster path needs Slurm; the Ryzen box has no such split.

---

## Part B — Run the pipeline on the Ryzen box

Everything below runs locally. No VPN, no Slurm.

### 1. Get the code

```bash
git clone https://github.com/johnnyhliang/reactor-data-harvest.git
cd reactor-data-harvest
pip install -r requirements.txt        # requests, pymupdf, pillow
```

### 2. Install Ollama (you have root here — the normal installer works)

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5vl:32b              # best quality; 128 GB handles it easily
# (qwen2.5vl:7b is a faster option for a first pass)
```

### 3. Harvest (downloads PDFs, extracts figure+caption pairs)

```bash
bash arc/harvest.sh                    # the harvest script works fine off-cluster
```

Writes `output_big/` (OSTI, ~12 captioned figs/PDF) and `output_patents/`
(~15 figs/patent) in HuggingFace imagefolder format. Re-runnable and resumable.

### 4. Filter + re-caption (the key step — runs directly, no Slurm)

```bash
ollama serve &                         # skip if it already runs as a service
python3 recaption.py --backend ollama --model qwen2.5vl:32b \
    --in output_big/metadata.jsonl --out output_big/metadata_filtered.jsonl
python3 recaption.py --backend ollama --model qwen2.5vl:32b \
    --in output_patents/metadata.jsonl --out output_patents/metadata_filtered.jsonl
```

Drops plots/charts/tangential photos and rewrites terse captions into rich ones.
**Resumable** — safe to stop/restart; rows already done are skipped. The printed
kept/dropped count is the **real usable-image number** (replaces the ~20–30%
estimate in `WRITEUP.md` §4).

### 5. The deliverable

```
output_big/metadata_filtered.jsonl        # OSTI, kept rows, enriched captions
output_patents/metadata_filtered.jsonl    # patents, same
```

Load it exactly like the paper's data. `imagefolder` looks for a file named
`metadata.jsonl`, so point it at a folder whose metadata is the filtered file:

```python
from datasets import load_dataset
ds = load_dataset("imagefolder", data_dir="output_big")
```

(To train on the filtered set, copy/rename `metadata_filtered.jsonl` →
`metadata.jsonl` in a fresh dir, or symlink it.)

### Scaling / remaining todos

See `HANDOFF.md` — add more queries to `arc/harvest.sh`, raise patent `--pages`
(pace it; Google throttles), dedup across documents, then hand the ablation
(baseline vs +harvest vs +recaptioned; measure KID/CMMD + failure count) to
whoever owns fine-tuning.
