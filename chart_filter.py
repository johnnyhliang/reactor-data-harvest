#!/usr/bin/env python3
"""Drop scientific charts without asking a model what a reactor looks like.

A chart here means a mostly empty framed plot (white axes area) or a dense
tick grid on a white margin. Labeled cutaways and hardware photos fail both
tests because their centers are full of ink. Author-caption plot words are
applied only when the center is also mostly empty, so a cutaway with a
swapped k-effective caption is kept.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

PLOT_CAPTION = re.compile(
    r"\b(?:plots?|graphs?|spectra|spectrum|versus|vs\.?|as a function of|"
    r"histograms?|sensitivity|uncertaint|dose rates?|reactivit|rod worth)\b",
    re.IGNORECASE,
)


def chart_features(path: Path) -> tuple[bool, float, float]:
    image = Image.open(path).convert("L")
    image.thumbnail((400, 400))
    pixels = np.asarray(image, dtype=np.float32)
    dark = pixels < 80
    row_ink = dark.mean(axis=1)
    col_ink = dark.mean(axis=0)
    row_lines = int((row_ink > 0.35).sum())
    col_lines = int((col_ink > 0.35).sum())
    height, width = pixels.shape
    edge = max(1, int(height * 0.12))
    side = max(1, int(width * 0.12))
    border = np.concatenate([
        pixels[:edge].ravel(), pixels[-edge:].ravel(),
        pixels[:, :side].ravel(), pixels[:, -side:].ravel(),
    ])
    center = pixels[int(height * 0.25):int(height * 0.75), int(width * 0.25):int(width * 0.75)]
    white_border = float((border > 240).mean())
    white_center = float((center > 240).mean())
    # A dense line count also matches CAD and meshes, so only the empty
    # framed plot is treated as a chart. row_lines/col_lines stay available
    # for debugging but do not decide.
    del row_lines, col_lines
    empty_frame = white_center > 0.85 and white_border > 0.70
    return empty_frame, white_center, white_border


def main() -> int:
    parser = argparse.ArgumentParser(description="Move chart-like figures out of an image folder")
    parser.add_argument("--images", type=Path, default=Path("sample_yield/images"))
    parser.add_argument("--metadata", type=Path, default=Path("sample_yield/scored_v2.jsonl"))
    parser.add_argument("--dropped", type=Path, default=Path("sample_yield/dropped"))
    parser.add_argument("--apply", action="store_true", help="move detected charts into --dropped")
    args = parser.parse_args()

    captions = {}
    if args.metadata.exists():
        for line in args.metadata.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                captions[Path(row["file_name"]).name] = row.get("text") or ""

    charts = []
    kept = []
    for path in sorted(args.images.glob("*")):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        is_chart, white_center, white_border = chart_features(path)
        caption = captions.get(path.name, "")
        caption_plot = bool(PLOT_CAPTION.search(caption)) and white_center > 0.55
        if is_chart or caption_plot:
            charts.append((path, "axes" if is_chart else "caption", white_center, white_border))
        else:
            kept.append(path.name)

    print(f"charts {len(charts)}  kept {len(kept)}  in {args.images}")
    for path, why, white_center, white_border in charts:
        print(f"  drop {why:7} wc={white_center:.2f} wb={white_border:.2f}  {path.name}")

    if args.apply and charts:
        args.dropped.mkdir(parents=True, exist_ok=True)
        for path, _, _, _ in charts:
            shutil.move(str(path), str(args.dropped / path.name))
        print(f"moved {len(charts)} -> {args.dropped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
