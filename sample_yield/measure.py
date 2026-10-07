"""Sample OSTI full text and score extracted figures with the AIMS CLIP filter.

Writes everything under sample_yield/. Does not call a paid API.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AIMS = ROOT / "aims-image-extraction"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(AIMS))

import harvest_osti  # noqa: E402
import parser as aims_parser  # noqa: E402
from PIL import Image  # noqa: E402
import io  # noqa: E402

OUT = ROOT / "sample_yield"
PDF_DIR = OUT / "pdfs"
IMG_DIR = OUT / "images"
QUERIES = [
    "PWR fuel assembly",
    "reactor core cross section",
    "control rod assembly",
    "TRISO fuel",
]
ROWS = 6  # reports requested per query; duplicates across queries are skipped
MAX_PDF_BYTES = 40 * 1024 * 1024


def download_capped(osti_id: str, dest: Path) -> str:
    """Return 'ok', 'no-pdf', or 'too-big'. Retries a broken download once."""
    import requests

    if dest.exists() and dest.stat().st_size > 1000 and dest.read_bytes()[:4] == b"%PDF":
        return "ok"
    url = harvest_osti.FULLTEXT.format(osti_id=osti_id)
    for attempt in (1, 2):
        try:
            r = requests.get(
                url,
                headers={"User-Agent": harvest_osti.HEADERS["User-Agent"]},
                timeout=(20, 60),
                allow_redirects=True,
                stream=True,
            )
        except requests.RequestException:
            if attempt == 2:
                return "no-pdf"
            time.sleep(2)
            continue
        if r.status_code != 200:
            r.close()
            return "no-pdf"
        chunks = []
        total = 0
        try:
            for chunk in r.iter_content(1024 * 256):
                if not chunk:
                    continue
                total += len(chunk)
                if total > MAX_PDF_BYTES:
                    r.close()
                    return "too-big"
                chunks.append(chunk)
        except requests.RequestException:
            r.close()
            if attempt == 2:
                return "no-pdf"
            time.sleep(2)
            continue
        r.close()
        data = b"".join(chunks)
        if not data.startswith(b"%PDF"):
            return "no-pdf"
        dest.write_bytes(data)
        return "ok"
    return "no-pdf"


def main() -> int:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    records_path = OUT / "records.jsonl"
    figures_path = OUT / "figures.jsonl"

    seen: set[str] = set()
    report_rows = []
    figure_rows = []

    for query in QUERIES:
        print(f"\n=== {query} ===", flush=True)
        try:
            found = harvest_osti.search_osti(query, ROWS)
        except Exception as exc:
            print(f"  search failed: {exc}", flush=True)
            continue
        for rec in found:
            osti_id = str(rec.get("osti_id") or "").strip()
            if not osti_id or osti_id in seen:
                continue
            seen.add(osti_id)
            title = harvest_osti._clean(rec.get("title") or "")
            product = rec.get("product_type") or ""
            has_link = any(
                (link.get("rel") == "fulltext") for link in (rec.get("links") or [])
            )
            row = {
                "osti_id": osti_id,
                "query": query,
                "title": title,
                "product_type": product,
                "doi": rec.get("doi") or "",
                "fulltext_link": has_link,
                "download": "skipped",
                "n_figures": 0,
            }
            if not has_link:
                row["download"] = "no-link"
                report_rows.append(row)
                print(f"  - {osti_id}: no fulltext link", flush=True)
                continue
            status = download_capped(osti_id, PDF_DIR / f"{osti_id}.pdf")
            row["download"] = status
            time.sleep(0.4)
            if status != "ok":
                report_rows.append(row)
                print(f"  - {osti_id}: {status}  [{title[:50]}]", flush=True)
                continue
            try:
                figs = list(harvest_osti.extract_figures(PDF_DIR / f"{osti_id}.pdf"))
            except Exception as exc:
                row["download"] = f"parse-error: {exc}"
                report_rows.append(row)
                print(f"  ! {osti_id}: {exc}", flush=True)
                continue
            row["n_figures"] = len(figs)
            for i, (data, ext, caption) in enumerate(figs):
                fname = f"osti_{osti_id}_{i:03d}.{ext}"
                (IMG_DIR / fname).write_bytes(data)
                figure_rows.append({
                    "file_name": f"images/{fname}",
                    "text": caption,
                    "osti_id": osti_id,
                    "report_title": title,
                    "query": query,
                    "product_type": product,
                    "doi": rec.get("doi") or "",
                })
            report_rows.append(row)
            print(f"  + {osti_id}: {len(figs):>3} figures  [{product}] {title[:50]}", flush=True)

    with records_path.open("w", encoding="utf-8") as fh:
        for row in report_rows:
            fh.write(json.dumps(row) + "\n")
    with figures_path.open("w", encoding="utf-8") as fh:
        for row in figure_rows:
            fh.write(json.dumps(row) + "\n")

    print(f"\nHarvested {len(figure_rows)} captioned figures from "
          f"{sum(r['download']=='ok' for r in report_rows)} PDFs "
          f"({len(report_rows)} unique records).", flush=True)
    if not figure_rows:
        return 0

    print("\nLoading CLIP...", flush=True)
    classifier = aims_parser.ImageClassifier(reference_dir=str(AIMS / "reference_images"))
    scored_path = OUT / "scored.jsonl"
    kept = 0
    with scored_path.open("w", encoding="utf-8") as fh:
        batch = 8
        for start in range(0, len(figure_rows), batch):
            chunk = figure_rows[start:start + batch]
            pil_images = []
            ok_idx = []
            for j, row in enumerate(chunk):
                path = OUT / row["file_name"]
                try:
                    pil_images.append(Image.open(io.BytesIO(path.read_bytes())).convert("RGB"))
                    ok_idx.append(j)
                except Exception as exc:
                    row = dict(row)
                    row["decision"] = False
                    row["reason"] = f"unreadable: {exc}"
                    fh.write(json.dumps(row) + "\n")
            embeddings = classifier.encode_batch([pil_images[k] for k in range(len(pil_images))]) if pil_images else None
            usable = 0
            for local_i, j in enumerate(ok_idx):
                result = classifier.classify_image(pil_images[local_i], embedding=embeddings[local_i])
                row = dict(chunk[j])
                row.update({
                    "decision": bool(result["decision"]),
                    "keep_score": result["keep_score"],
                    "reject_score": result["reject_score"],
                    "keep_label": result["keep_label"],
                    "reject_label": result["reject_label"],
                    "reason": result["reason"],
                })
                if row["decision"]:
                    kept += 1
                fh.write(json.dumps(row) + "\n")
                usable += 1
            print(f"  scored {min(start + batch, len(figure_rows))}/{len(figure_rows)}  kept so far {kept}", flush=True)

    print(f"\nCLIP kept {kept} / {len(figure_rows)}", flush=True)
    print(f"Wrote {scored_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
