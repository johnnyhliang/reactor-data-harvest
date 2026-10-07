#!/usr/bin/env python3
"""
recaption.py -- Stage 5: filter low-quality figures + enrich captions with a VLM.

The harvester pulls EVERY captioned figure, including data plots/charts and
tangential site photos. This stage uses a vision-language model to:
  1. CLASSIFY  -- keep real nuclear objects (schematic/cutaway/cross-section/
                  CAD/component photo/facility layout); drop plots/charts/tables.
  2. RE-CAPTION-- rewrite the terse source caption into a rich, visually-grounded
                  one (fixes the paper's "captions too short" limitation) WITHOUT
                  collecting any new image.

BACKENDS (auto-detected, in order):
  - ollama    : local, free, runs on your Ryzen AI box. Set OLLAMA_HOST if remote.
  - anthropic : cloud API, needs ANTHROPIC_API_KEY.
  - passthrough: no backend reachable -> copies everything unchanged (plumbing
                 test only; keeps all rows, no filtering). This is what runs on a
                 low-power machine with no model available.

Recommended on the Ryzen box:
    ollama pull llama3.2-vision:11b        # or qwen2-vl / llava
    python3 recaption.py --backend ollama --model llama3.2-vision:11b \\
        --in output_big/metadata.jsonl --out output_big/metadata_filtered.jsonl

The directory and captions are scanned before a VLM is contacted. Obvious caption
rejects are cheap; only remaining candidates use the model. Kept rows go to --out,
while rejects and model failures go to --out.rejected.jsonl. Both files are
resumable, so a stopped local pass does not repeat completed model calls. Use
--scan-only to inspect inventory and caption-only rejects without running a model.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
from pathlib import Path

try:
    import requests
except ImportError:  # scan-only and caption prefilter do not need network access
    requests = None

SYSTEM_PROMPT = """You curate images for a nuclear-energy figure dataset. Be highly selective.
The preferred image is a photograph of a recognizable nuclear-energy object or
hardware (fuel, reactor vessel/core, control rods, casks, instruments, or related
equipment). A schematic is acceptable only when it clearly depicts a meaningful,
recognizable nuclear engineering concept such as a reactor cross-section or fuel
assembly. A detailed, labeled nuclear-facility concept may qualify as a separate
schematic category. Require a clear connection to nuclear energy; do not infer
relevance from the report topic alone.

DROP (keep=false): every 2D or 3D plot, graph, bar/chart, histogram, contour plot
or map of values; flow charts and process/data-flow diagrams; tables; equations;
generic maps, site plans, facility overviews, aerial views, buildings without
visible relevant hardware; and diagrams that are abstract, cluttered, ambiguous,
or not meaningful even to a field expert. When uncertain, drop it. Never keep an
image merely because its caption contains nuclear terminology.

Return STRICT JSON only, no prose:
{"keep": true|false, "category": "component_photo|schematic|cutaway|cross_section|facility_schematic|plot|chart|contour|flowchart|table|other", "caption": "rich 1-2 sentence visually-grounded description; preserve technical terms from the original; describe geometry and viewpoint"}"""

# These words identify obvious non-targets from captions alone, before image bytes
# are read or a vision model is called. This is intentionally a reject-only pass.
CAPTION_JUNK_RE = re.compile(
    r"\b(?:bar\s+charts?|histograms?|contour\s+(?:plots?|maps?)|(?:2|3)[- ]?d\s+plots?|"
    r"scatter\s+plots?|flow\s*charts?|data\s+flows?|flow\s+diagrams?|"
    r"map\s+layouts?|maps?\s+of\s+(?:values|data|inventory)|"
    r"(?:line|area|surface)\s+plot|(?:temperature|burnup|pressure|power)\s+"
    r"(?:versus|vs\.?|as\s+a\s+function\s+of)|"
    r"plots?|graphs?|spectra|spectrum|sensitivity|uncertaint(?:y|ies)|"
    r"dose rates?|reactivit(?:y|ies)|rod worth)\b", re.IGNORECASE
)


def _b64(path: Path) -> str:
    return base64.standard_b64encode(path.read_bytes()).decode()


def _parse_json(text: str) -> dict | None:
    """Extract the first JSON object from a possibly-noisy model reply."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):]
    start, depth = text.find("{"), 0
    if start < 0:
        return None
    for i in range(start, len(text)):
        depth += text[i] == "{"
        depth -= text[i] == "}"
        if depth == 0:
            try:
                return json.loads(text[start:i + 1])
            except json.JSONDecodeError:
                return None
    return None


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
def ollama_available(host: str) -> bool:
    if requests is None:
        return False
    try:
        return requests.get(f"{host}/api/tags", timeout=3).status_code == 200
    except requests.RequestException:
        return False


def call_ollama(host: str, model: str, image_path: Path, original: str) -> dict | None:
    payload = {
        "model": model,
        "format": "json",          # force JSON output
        "stream": False,
        "options": {"temperature": 0},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",
             "content": f"Original caption: {original}",
             "images": [_b64(image_path)]},
        ],
    }
    try:
        r = requests.post(f"{host}/api/chat", json=payload, timeout=300)
        r.raise_for_status()
        return _parse_json(r.json()["message"]["content"])
    except (requests.RequestException, KeyError) as e:
        print(f"  ! ollama error on {image_path.name}: {e}", file=sys.stderr)
        return None


def call_anthropic(model: str, image_path: Path, original: str) -> dict | None:
    try:
        import anthropic
    except ImportError:
        print("  ! `pip install anthropic` to use the anthropic backend", file=sys.stderr)
        return None
    media = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
    client = anthropic.Anthropic()
    try:
        msg = client.messages.create(
            model=model, max_tokens=400, system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64",
                 "media_type": media, "data": _b64(image_path)}},
                {"type": "text", "text": f"Original caption: {original}"},
            ]}],
        )
        return _parse_json(msg.content[0].text)
    except Exception as e:
        print(f"  ! anthropic error on {image_path.name}: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
def resolve_backend(requested: str, ollama_host: str) -> str:
    if requested != "auto":
        return requested
    if ollama_available(ollama_host):
        return "ollama"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    return "passthrough"


def already_done(out_path: Path) -> set[str]:
    done = set()
    if out_path.exists():
        for line in out_path.open():
            try:
                done.add(json.loads(line)["file_name"])
            except (json.JSONDecodeError, KeyError):
                pass
    return done


def scan_metadata(inp: Path, root: Path) -> tuple[list[dict], int, int, int]:
    """Read the directory manifest once and reject missing/obvious junk cheaply."""
    records = []
    missing = junk = duplicates = 0
    seen: set[str] = set()
    with inp.open(encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            rec = json.loads(line)
            name = rec.get("file_name", "")
            if name in seen:
                duplicates += 1
                continue
            seen.add(name)
            image = root / name
            if not image.is_file():
                missing += 1
                continue
            if CAPTION_JUNK_RE.search(rec.get("text", "")):
                junk += 1
                rec["filter_reason"] = "caption_prefilter"
                records.append((rec, image, False))
            else:
                records.append((rec, image, True))
    return records, missing, junk, duplicates


def main() -> int:
    ap = argparse.ArgumentParser(description="VLM filter + re-caption harvested figures")
    ap.add_argument("--in", dest="inp", type=Path, default=Path("output/metadata.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("output/metadata_filtered.jsonl"))
    ap.add_argument("--root", type=Path, default=None,
                    help="dir that file_name paths are relative to (default: --in's parent)")
    ap.add_argument("--backend", choices=["auto", "ollama", "anthropic", "passthrough"],
                    default="auto")
    ap.add_argument("--model", default="llama3.2-vision:11b",
                    help="ollama model tag, or an anthropic model id (e.g. claude-sonnet-5)")
    ap.add_argument("--ollama-host", default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    ap.add_argument("--limit", type=int, default=0, help="process at most N rows (0 = all)")
    ap.add_argument("--scan-only", action="store_true",
                    help="inventory images and apply caption-only prefilter; never call a VLM")
    args = ap.parse_args()

    root = args.root or args.inp.parent
    records, missing, prefiltered, duplicates = scan_metadata(args.inp, root)
    candidates = sum(eligible for _, _, eligible in records)
    print(f"directory scan: {len(records)} unique images, {candidates} candidates, "
          f"{prefiltered} obvious caption rejects, {missing} missing, {duplicates} duplicates")
    if args.scan_only:
        return 0

    backend = resolve_backend(args.backend, args.ollama_host)
    print(f"backend: {backend}"
          + (f"  model: {args.model}" if backend != "passthrough" else "")
          + (f"  host: {args.ollama_host}" if backend == "ollama" else ""))
    if backend == "passthrough":
        print("  (no VLM reachable -> copying rows unchanged, NO filtering. "
              "Run on the Ryzen box with Ollama for a real pass.)")

    done = already_done(args.out)
    if done:
        print(f"  resuming: {len(done)} rows already in {args.out}")

    rejects_path = args.out.with_suffix(args.out.suffix + ".rejected.jsonl")
    done.update(already_done(rejects_path))
    kept = dropped = failed = 0
    cats: dict[str, int] = {}
    mode = "a" if done else "w"
    reject_mode = "a" if done else "w"
    with args.out.open(mode, encoding="utf-8") as fout, rejects_path.open(reject_mode, encoding="utf-8") as freject:
        for rec, img, eligible in records:
            if args.limit and (kept + dropped + failed) >= args.limit:
                break
            if rec["file_name"] in done:
                continue
            if not eligible:
                freject.write(json.dumps(rec) + "\n")
                freject.flush()
                done.add(rec["file_name"])
                dropped += 1
                continue

            if backend == "passthrough":
                result = {"keep": True, "category": "unknown", "caption": rec["text"]}
            elif backend == "ollama":
                result = call_ollama(args.ollama_host, args.model, img, rec["text"])
            else:
                result = call_anthropic(args.model, img, rec["text"])

            if result is None:              # model/parse failure -> keep, flag for review
                failed += 1
                rec["review"] = "vlm_failed"
                rec["filter_reason"] = "vlm_failed"
                freject.write(json.dumps(rec) + "\n")
                freject.flush()
                done.add(rec["file_name"])
                continue

            cat = result.get("category", "unknown")
            cats[cat] = cats.get(cat, 0) + 1
            if not result.get("keep"):
                dropped += 1
                rec["filter_reason"] = "vlm_rejected"
                rec["category"] = cat
                freject.write(json.dumps(rec) + "\n")
                freject.flush()
                done.add(rec["file_name"])
                continue
            rec["text"] = result.get("caption", rec["text"])
            rec["category"] = cat
            rec["caption_source"] = "vlm-enriched"
            fout.write(json.dumps(rec) + "\n")
            fout.flush()                    # resumable: durable after each row
            done.add(rec["file_name"])
            kept += 1
            if (kept + dropped) % 25 == 0:
                print(f"  ...{kept} kept / {dropped} dropped")

    print("\n" + "=" * 50)
    print(f"kept {kept}  dropped {dropped}  failed {failed}  -> {args.out}")
    if cats:
        print("categories seen:")
        for c, n in sorted(cats.items(), key=lambda x: -x[1]):
            print(f"  {n:>4}  {c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
