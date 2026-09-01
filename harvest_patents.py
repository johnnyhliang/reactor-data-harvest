#!/usr/bin/env python3
"""
harvest_patents.py -- Harvest captioned nuclear DRAWINGS from patents.

US patents are public domain. Every patent contains labeled engineering drawings
(FIG. 1, FIG. 2A, ...) AND a "Brief Description of the Drawings" section that
describes each figure in a full sentence -- e.g. "FIG. 1 is a schematic
cross-sectional view of a control drum for a mobile nuclear reactor." That
sentence is a rich, ready-made caption, far more descriptive than the terse
figure captions in reports/journals. This complements the OSTI harvest with a
large second public-domain source.

Structure this exploits (verified on real patents):
  - Each drawing sheet is its own PDF page: a "FIG. N" text label + one full-page
    image (the rasterized sheet).
  - The body text (later pages) carries the per-figure descriptions.
  So we: parse descriptions from the body, then for each drawing page extract the
  sheet image and attach the description of the FIG number(s) printed on it.

Discovery uses Google Patents' keyless XHR endpoint (no API key required).

CAVEAT baked into the data: patent figures are black-and-white LINE DRAWINGS with
reference numerals -- a narrow visual style. They add geometric/schematic
diversity and excellent captions, but are not photorealistic. Flag this when
mixing into a training set.

No GPU required.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests

try:
    import pymupdf
except ImportError:
    import fitz as pymupdf

XHR = "https://patents.google.com/xhr/query"
PDF_HOST = "https://patentimages.storage.googleapis.com/"
UA = {"User-Agent": "Mozilla/5.0 (research prototype reactor-data-harvest/0.1)"}

# "FIG. 1", "FIG . 2A", "FIGS. 3-4" -- patents render odd spacing, so be lenient.
FIG_LABEL = re.compile(r"FIG\s*S?\s*\.?\s*(\d+[A-Z]?)", re.IGNORECASE)
# A description sentence: "FIG. 1 is a ...", "FIG. 2A illustrates ...".
FIG_DESC = re.compile(
    r"FIG\s*\.?\s*(\d+[A-Z]?)\s+"
    r"(is|are|shows?|depicts?|illustrates?|provides?|presents?|represents?|"
    r"schematically|comprises?)\b[^.]*\.",
    re.IGNORECASE,
)


def search_patents(query: str, pages: int = 1, before: str | None = None,
                   cpc: str | None = "G21C") -> list[dict]:
    """
    Return patent hits (dicts with pdf path, title, number) via Google Patents XHR.

    `cpc` restricts to a patent classification. G21C = "Nuclear reactors" -- this
    is essential: a bare keyword search for "control rod drive" otherwise returns
    surgical-robot patents. G21 is the broader "nuclear physics" section.
    """
    out: list[dict] = []
    for pg in range(pages):
        # Build the inner query, then percent-encode it EXACTLY ONCE as the value
        # of `url`. (Passing a params dict double-encodes -> %2520 -> 503.)
        inner = f"q={query}"
        if cpc:
            inner += f"&cpc={cpc}"
        if before:
            inner += f"&before=priority:{before}"
        if pg:
            inner += f"&page={pg}"
        full = f"{XHR}?url={requests.utils.quote(inner, safe='')}&exp="
        data = None
        for attempt in range(4):                       # backoff on 503 rate-limits
            try:
                r = requests.get(full, headers=UA, timeout=30)
                if r.status_code == 503:
                    time.sleep(2 * (attempt + 1))
                    continue
                r.raise_for_status()
                data = r.json()
                break
            except (requests.RequestException, ValueError) as e:
                if attempt == 3:
                    print(f"  ! search failed p{pg}: {e}", file=sys.stderr)
        if data is None:
            break
        clusters = data.get("results", {}).get("cluster", [])
        for cl in clusters:
            for item in cl.get("result", []):
                pat = item.get("patent", {})
                if pat.get("pdf"):
                    out.append({
                        "number": pat.get("publication_number", ""),
                        "title": " ".join((pat.get("title") or "").split()),
                        "pdf": pat["pdf"],
                    })
        time.sleep(0.6)
    return out


def download_pdf(pdf_path: str, dest: Path) -> Path | None:
    try:
        r = requests.get(PDF_HOST + pdf_path, headers=UA, timeout=60)
    except requests.RequestException:
        return None
    if r.status_code != 200 or r.content[:4] != b"%PDF":
        return None
    dest.write_bytes(r.content)
    return dest


def parse_descriptions(doc) -> dict[str, str]:
    """Map figure key ('1', '2A', ...) -> full description sentence from body text."""
    full = " ".join(p.get_text() for p in doc)
    full = re.sub(r"\s+", " ", full)
    desc: dict[str, str] = {}
    for m in FIG_DESC.finditer(full):
        key = m.group(1).upper()
        sentence = m.group(0).strip()
        if key not in desc or len(sentence) > len(desc[key]):
            desc[key] = sentence
    return desc


def is_drawing_page(page) -> bool:
    """A drawing sheet: short text (just labels), at least one sizable image."""
    txt = page.get_text().strip()
    if len(txt) > 700:            # body/claims pages have lots of text
        return False
    return bool(FIG_LABEL.search(txt)) and bool(page.get_images())


def extract_patent_figures(pdf_path: Path, min_px: int = 200, title: str = ""):
    """Yield (image_bytes, ext, caption, fig_key) for each drawing sheet."""
    doc = pymupdf.open(pdf_path)
    descs = parse_descriptions(doc)
    for page in doc:
        if not is_drawing_page(page):
            continue
        figs = [m.group(1).upper() for m in FIG_LABEL.finditer(page.get_text())]
        figs = list(dict.fromkeys(figs))  # unique, order-preserving
        # take the largest embedded image on the sheet
        best = None
        for img in page.get_images(full=True):
            xref = img[0]
            rects = page.get_image_rects(xref)
            if not rects:
                continue
            area = rects[0].width * rects[0].height
            if min(rects[0].width, rects[0].height) < min_px:
                continue
            if best is None or area > best[1]:
                best = (xref, area)
        if best is None:
            continue
        info = doc.extract_image(best[0])
        # caption: prefer the parsed description(s); fall back to a label
        sentences = [descs[f] for f in figs if f in descs]
        if sentences:
            caption = " ".join(sentences)
        elif figs and title:
            # fallback: ground the drawing in the patent's subject
            caption = f"FIG. {figs[0]}: engineering drawing from patent \"{title}\"."
        else:
            caption = f"FIG. {figs[0]} engineering drawing." if figs else "Patent drawing."
        yield info["image"], info["ext"], caption, (figs[0] if figs else "?")
    doc.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="Harvest captioned nuclear drawings from patents")
    ap.add_argument("--queries", nargs="+", default=[
        "nuclear reactor fuel assembly", "control rod drive mechanism",
        "nuclear fuel rod cladding", "spent nuclear fuel storage cask",
    ])
    ap.add_argument("--pages", type=int, default=1, help="result pages per query (~10 patents/page)")
    ap.add_argument("--cpc", default="G21C",
                    help="patent classification filter; G21C=nuclear reactors (empty '' to disable)")
    ap.add_argument("--before", default="20240101", help="priority date cutoff (older = clearly public)")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "output_patents")
    ap.add_argument("--keep-pdfs", action="store_true")
    args = ap.parse_args()

    img_dir = args.out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir = args.out / "pdfs"; pdf_dir.mkdir(exist_ok=True)
    meta_path = args.out / "metadata.jsonl"

    n_pat = n_pdf = n_fig = 0
    seen: set[str] = set()
    per_query: dict[str, int] = {}

    with meta_path.open("w") as meta:
        for q in args.queries:
            print(f"\n=== query: {q!r} ===", flush=True)
            hits = search_patents(q, pages=args.pages, before=args.before,
                                  cpc=(args.cpc or None))
            qf = 0
            for h in hits:
                num = h["number"]
                if not num or num in seen:
                    continue
                seen.add(num); n_pat += 1
                pdf_path = pdf_dir / f"{num}.pdf"
                if download_pdf(h["pdf"], pdf_path) is None:
                    print(f"  - {num}: no PDF"); continue
                n_pdf += 1
                try:
                    figs = list(extract_patent_figures(pdf_path, title=h["title"]))
                except Exception as e:
                    print(f"  ! {num}: parse error {e}", file=sys.stderr); figs = []
                for i, (data, ext, caption, key) in enumerate(figs):
                    fname = f"patent_{num}_{i:03d}.{ext}"
                    (img_dir / fname).write_bytes(data)
                    meta.write(json.dumps({
                        "file_name": f"images/{fname}",
                        "text": caption,
                        "source": "patent",
                        "patent_number": num,
                        "patent_title": h["title"],
                        "fig": key,
                        "query": q,
                    }) + "\n")
                    n_fig += 1; qf += 1
                print(f"  + {num}: {len(figs):>2} figures  [{h['title'][:55]}]")
                if not args.keep_pdfs:
                    pdf_path.unlink(missing_ok=True)
            per_query[q] = qf

    print("\n" + "=" * 60)
    print("PATENT HARVEST REPORT")
    print("=" * 60)
    print(f"  patents seen     : {n_pat}")
    print(f"  PDFs downloaded  : {n_pdf}")
    print(f"  captioned figures: {n_fig}")
    if n_pdf:
        print(f"  yield            : {n_fig / n_pdf:.1f} figures / patent")
    for q, c in per_query.items():
        print(f"    {c:>4}  {q}")
    print(f"\n  -> {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
