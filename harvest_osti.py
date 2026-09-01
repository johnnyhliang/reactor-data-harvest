#!/usr/bin/env python3
"""
harvest_osti.py -- Harvest real, captioned nuclear figures from OSTI.gov.

OSTI.gov is the U.S. Department of Energy's technical-report repository. Reports
authored under DOE contracts are U.S. government works (public domain), full-text
PDFs are openly downloadable, and every figure carries an author-written caption.
That makes it a large, license-clean, already-captioned source of real nuclear
imagery -- exactly the gap the NuclearDiffusion paper (arXiv:2608.04030) hit when
it could only recover ~1,000 images.

This script:
  1. Searches OSTI for reports matching nuclear-component queries.
  2. Downloads each report's full-text PDF.
  3. Extracts (figure image, caption) pairs by spatially pairing each raster
     image with the nearest "Figure N ..." caption block on the same page.
  4. Writes them in HuggingFace imagefolder format (images/ + metadata.jsonl),
     the same format the paper's fine-tuning pipeline consumes.

This is a feasibility prototype: it proves the source and the extraction work,
and reports how many captioned images each query yields. Quality filtering
(dropping plots/charts, keeping true schematics/photos) is a separate downstream
stage -- see recaption.py and the README.

No GPU required. Pure HTTP + PDF parsing.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
from pathlib import Path

import requests

try:
    import pymupdf  # PyMuPDF >= 1.24 exposes the `pymupdf` name
except ImportError:  # older wheels only expose `fitz`
    import fitz as pymupdf

OSTI_SEARCH = "https://www.osti.gov/api/v1/records"
FULLTEXT = "https://www.osti.gov/servlets/purl/{osti_id}"
HEADERS = {"Accept": "application/json", "User-Agent": "reactor-data-harvest/0.1 (research prototype)"}

# A caption block starts with "Figure 1", "Fig. 2-3", "FIGURE 4", etc.
CAP_RE = re.compile(r"^\s*(fig(?:ure)?\.?\s*[\d]+(?:[-.–]\d+)?)\b", re.IGNORECASE)
# In-text references ("Figure 3-9 compares ...", "Figure 2 shows ...") are NOT
# captions. A real caption is followed by a title, not a finite verb clause.
INTEXT_RE = re.compile(
    r"^\s*fig(?:ure)?\.?\s*[\d.–-]+\s+"
    r"(shows?|compares?|presents?|illustrates?|depicts?|gives?|lists?|summari[sz]es?|"
    r"provides?|displays?|indicates?|reports?|is\b|are\b|was\b|were\b|can\b|above|below)",
    re.IGNORECASE,
)


def search_osti(query: str, rows: int, max_pages: int = 1) -> list[dict]:
    """Return OSTI records (dicts) matching `query`."""
    records: list[dict] = []
    for page in range(1, max_pages + 1):
        params = {"q": query, "rows": rows, "page": page}
        r = requests.get(OSTI_SEARCH, params=params, headers=HEADERS, timeout=30)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        records.extend(batch)
        time.sleep(0.5)  # be polite to the API
    return records


def download_pdf(osti_id: str, dest: Path, timeout: int = 60) -> Path | None:
    """Download the full-text PDF for an OSTI record; return path or None."""
    url = FULLTEXT.format(osti_id=osti_id)
    try:
        r = requests.get(url, headers={"User-Agent": HEADERS["User-Agent"]},
                         timeout=timeout, allow_redirects=True)
    except requests.RequestException:
        return None
    if r.status_code != 200 or not r.content[:4] == b"%PDF":
        return None
    dest.write_bytes(r.content)
    return dest


def _clean(text: str) -> str:
    return " ".join(text.split())


def extract_figures(pdf_path: Path, min_px: int = 120, max_caption_chars: int = 500):
    """
    Yield (image_bytes, ext, caption) for each figure in the PDF.

    Pairing rule: for each caption block, attach the image whose bounding box is
    vertically closest to it on the same page (captions usually sit directly
    below, sometimes above, their figure).
    """
    doc = pymupdf.open(pdf_path)
    for page in doc:
        blocks = page.get_text("blocks")  # (x0,y0,x1,y1,text,block_no,block_type)
        captions = []
        for b in blocks:
            txt = _clean(b[4])
            if not CAP_RE.match(txt):
                continue
            if INTEXT_RE.match(txt):
                continue  # in-text reference, not a caption
            if len(txt) > max_caption_chars or len(txt) < 12:
                continue
            cy = (b[1] + b[3]) / 2
            captions.append((cy, txt))
        if not captions:
            continue

        # Collect image xrefs with their placement rectangles on this page.
        placed = []
        for img in page.get_images(full=True):
            xref = img[0]
            rects = page.get_image_rects(xref)
            if not rects:
                continue
            rect = rects[0]
            if min(rect.width, rect.height) < min_px:
                continue  # skip logos / rules / tiny glyphs
            placed.append(((rect.y0 + rect.y1) / 2, xref))
        if not placed:
            continue

        used = set()
        for cy, cap in captions:
            # nearest unused image by vertical distance
            best = min(
                (p for p in placed if p[1] not in used),
                key=lambda p: abs(p[0] - cy),
                default=None,
            )
            if best is None:
                continue
            _, xref = best
            used.add(xref)
            info = doc.extract_image(xref)
            yield info["image"], info["ext"], cap
    doc.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="Harvest captioned nuclear figures from OSTI.gov")
    ap.add_argument("--queries", nargs="+", default=[
        "PWR fuel assembly", "reactor core cross section",
        "control rod assembly", "spent fuel cask", "TRISO fuel",
    ], help="OSTI search queries (each yields a batch of reports)")
    ap.add_argument("--rows", type=int, default=10, help="reports per query")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "output")
    ap.add_argument("--keep-pdfs", action="store_true", help="retain downloaded PDFs")
    args = ap.parse_args()

    img_dir = args.out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir = args.out / "pdfs"
    pdf_dir.mkdir(exist_ok=True)
    meta_path = args.out / "metadata.jsonl"

    n_reports = n_pdfs = n_figs = 0
    seen_ids: set[str] = set()
    per_query: dict[str, int] = {}

    with meta_path.open("w") as meta:
        for q in args.queries:
            print(f"\n=== query: {q!r} ===", flush=True)
            try:
                records = search_osti(q, args.rows)
            except requests.RequestException as e:
                print(f"  ! search failed: {e}", file=sys.stderr)
                continue
            q_figs = 0
            for rec in records:
                osti_id = str(rec.get("osti_id") or "").strip()
                if not osti_id or osti_id in seen_ids:
                    continue
                seen_ids.add(osti_id)
                n_reports += 1
                title = _clean(rec.get("title") or "")
                pdf_path = pdf_dir / f"{osti_id}.pdf"
                if download_pdf(osti_id, pdf_path) is None:
                    print(f"  - {osti_id}: no fulltext PDF")
                    continue
                n_pdfs += 1
                try:
                    figs = list(extract_figures(pdf_path))
                except Exception as e:  # keep harvesting other PDFs
                    print(f"  ! {osti_id}: parse error {e}", file=sys.stderr)
                    figs = []
                for i, (data, ext, caption) in enumerate(figs):
                    fname = f"osti_{osti_id}_{i:03d}.{ext}"
                    (img_dir / fname).write_bytes(data)
                    meta.write(json.dumps({
                        "file_name": f"images/{fname}",
                        "text": caption,
                        "source": "osti",
                        "osti_id": osti_id,
                        "report_title": title,
                        "query": q,
                    }) + "\n")
                    n_figs += 1
                    q_figs += 1
                print(f"  + {osti_id}: {len(figs):>2} figures  [{title[:60]}]")
                if not args.keep_pdfs:
                    pdf_path.unlink(missing_ok=True)
            per_query[q] = q_figs

    print("\n" + "=" * 60)
    print("FEASIBILITY REPORT")
    print("=" * 60)
    print(f"  reports searched : {n_reports}")
    print(f"  PDFs downloaded  : {n_pdfs}")
    print(f"  captioned figures: {n_figs}")
    if n_pdfs:
        print(f"  yield            : {n_figs / n_pdfs:.1f} figures / PDF")
    print("  per query:")
    for q, c in per_query.items():
        print(f"    {c:>4}  {q}")
    print(f"\n  -> {meta_path}")
    print("  Load with: datasets.load_dataset('imagefolder', data_dir='output')")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
