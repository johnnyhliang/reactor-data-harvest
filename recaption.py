#!/usr/bin/env python3
"""
recaption.py -- Stage 5: filter (plots vs. schematics) + enrich captions with a VLM.

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

The run is RESUMABLE: rows already present in --out are skipped, so a slow local
pass can be stopped and restarted.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path

import requests

SYSTEM_PROMPT = """You label figures for a nuclear-engineering image dataset used to train a text-to-image model.
Given a figure image and its original caption, decide if it depicts a PHYSICAL nuclear object worth keeping.

KEEP (keep=true): reactor schematics, cross-sections, cutaways, CAD renders, fuel
assembly/rod/pellet diagrams, cask/vessel/core layouts, component photographs,
facility layout diagrams.
DROP (keep=false): line/bar/scatter plots, contour maps of data, histograms,
tables, pure equations, flow charts, and photos not showing a nuclear component
(e.g. generic buildings, rail track, aerial site views with no visible hardware).

Return STRICT JSON only, no prose:
{"keep": true|false, "category": "schematic|cutaway|cross_section|cad|component_photo|layout|plot|chart|table|flowchart|other", "caption": "rich 1-2 sentence visually-grounded description; preserve technical terms from the original; describe geometry and viewpoint"}"""


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
    args = ap.parse_args()

    root = args.root or args.inp.parent
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

    kept = dropped = failed = 0
    cats: dict[str, int] = {}
    mode = "a" if done else "w"
    with args.inp.open() as fin, args.out.open(mode) as fout:
        for n, line in enumerate(fin):
            if args.limit and (kept + dropped + failed) >= args.limit:
                break
            rec = json.loads(line)
            if rec["file_name"] in done:
                continue
            img = root / rec["file_name"]
            if not img.exists():
                failed += 1
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
                fout.write(json.dumps(rec) + "\n")
                fout.flush()
                continue

            cat = result.get("category", "unknown")
            cats[cat] = cats.get(cat, 0) + 1
            if not result.get("keep"):
                dropped += 1
                continue
            rec["text"] = result.get("caption", rec["text"])
            rec["category"] = cat
            rec["caption_source"] = "vlm-enriched"
            fout.write(json.dumps(rec) + "\n")
            fout.flush()                    # resumable: durable after each row
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
