#!/usr/bin/env python3
"""Download more OSTI technical reports and keep non-chart figures."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from chart_filter import PLOT_CAPTION, chart_features
from harvest_osti import _clean, download_pdf, extract_figures, search_osti

ROOT = Path(__file__).parent
OUT = ROOT / "more_batch"
QUERIES = [
    "TRISO fuel",
    "fuel assembly",
    "spent fuel cask",
    "reactor pressure vessel",
    "control rod drive",
    "pebble bed reactor",
]
TARGET_PDFS = 70
ROWS = 25
PAGES = 4


def known_ids() -> set[str]:
    ids = set()
    for folder in (ROOT / "sample_yield" / "pdfs", OUT / "pdfs"):
        if folder.exists():
            ids.update(p.stem for p in folder.glob("*.pdf"))
    return ids


def main() -> int:
    img_dir = OUT / "images"
    drop_dir = OUT / "dropped"
    pdf_dir = OUT / "pdfs"
    for folder in (img_dir, drop_dir, pdf_dir):
        folder.mkdir(parents=True, exist_ok=True)
    seen = known_ids()
    meta_path = OUT / "metadata.jsonl"
    kept = dropped = pdfs = 0
    with meta_path.open("a", encoding="utf-8") as meta:
        for query in QUERIES:
            if pdfs >= TARGET_PDFS:
                break
            print(f"\n=== {query} ===", flush=True)
            try:
                records = search_osti(query, ROWS, max_pages=PAGES)
            except Exception as exc:
                print(f"  search failed: {exc}", file=sys.stderr)
                continue
            for rec in records:
                if pdfs >= TARGET_PDFS:
                    break
                if rec.get("product_type") != "Technical Report":
                    continue
                osti_id = str(rec.get("osti_id") or "").strip()
                if not osti_id or osti_id in seen:
                    continue
                seen.add(osti_id)
                pdf_path = pdf_dir / f"{osti_id}.pdf"
                if download_pdf(osti_id, pdf_path, timeout=90) is None:
                    print(f"  - {osti_id}: no pdf", flush=True)
                    continue
                pdfs += 1
                title = _clean(rec.get("title") or "")
                try:
                    figs = list(extract_figures(pdf_path))
                except Exception as exc:
                    print(f"  ! {osti_id}: {exc}", file=sys.stderr)
                    figs = []
                n_keep = 0
                for i, (data, ext, caption) in enumerate(figs):
                    fname = f"osti_{osti_id}_{i:03d}.{ext}"
                    path = img_dir / fname
                    path.write_bytes(data)
                    is_chart, white_center, _ = chart_features(path)
                    caption_plot = bool(PLOT_CAPTION.search(caption)) and white_center > 0.55
                    label = "drop" if is_chart or caption_plot else "keep"
                    if label == "drop":
                        path.rename(drop_dir / fname)
                        dropped += 1
                    else:
                        n_keep += 1
                        kept += 1
                    meta.write(json.dumps({
                        "file_name": f"images/{fname}",
                        "text": caption,
                        "osti_id": osti_id,
                        "report_title": title,
                        "query": query,
                        "label": label,
                    }) + "\n")
                print(f"  + {osti_id}: {n_keep}/{len(figs)} kept  [{title[:70]}]", flush=True)
    print(f"\npdfs={pdfs} kept={kept} dropped={dropped}", flush=True)
    print(f"images: {img_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
