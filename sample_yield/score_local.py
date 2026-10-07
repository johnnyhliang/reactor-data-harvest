"""Score figures already downloaded under sample_yield/pdfs. No new PDF downloads."""
from __future__ import annotations

import io
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AIMS = ROOT / "aims-image-extraction"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(AIMS))

import harvest_osti  # noqa: E402
import parser as aims_parser  # noqa: E402
from PIL import Image  # noqa: E402

OUT = ROOT / "sample_yield"
PDF_DIR = OUT / "pdfs"
IMG_DIR = OUT / "images"


def record_meta(osti_id: str) -> dict:
    url = f"https://www.osti.gov/api/v1/records/{osti_id}"
    req = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": harvest_osti.HEADERS["User-Agent"]},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8", "replace"))
    except Exception as exc:
        return {"title": "", "product_type": "", "doi": "", "error": str(exc)}
    if isinstance(data, list):
        data = data[0] if data else {}
    return {
        "title": harvest_osti._clean(data.get("title") or ""),
        "product_type": data.get("product_type") or "",
        "doi": data.get("doi") or "",
    }


def main() -> int:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    figure_rows = []
    report_rows = []
    for pdf in pdfs:
        osti_id = pdf.stem
        meta = record_meta(osti_id)
        time.sleep(0.3)
        try:
            figs = list(harvest_osti.extract_figures(pdf))
        except Exception as exc:
            report_rows.append({"osti_id": osti_id, "n_figures": 0, "error": str(exc), **meta})
            print(f"! {osti_id}: {exc}", flush=True)
            continue
        report_rows.append({"osti_id": osti_id, "n_figures": len(figs), **meta})
        print(f"+ {osti_id}: {len(figs):>3}  [{meta.get('product_type')}] {meta.get('title','')[:60]}", flush=True)
        for i, (data, ext, caption) in enumerate(figs):
            fname = f"osti_{osti_id}_{i:03d}.{ext}"
            (IMG_DIR / fname).write_bytes(data)
            figure_rows.append({
                "file_name": f"images/{fname}",
                "text": caption,
                "osti_id": osti_id,
                "report_title": meta.get("title", ""),
                "product_type": meta.get("product_type", ""),
                "doi": meta.get("doi", ""),
            })

    (OUT / "records.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in report_rows), encoding="utf-8"
    )
    (OUT / "figures.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in figure_rows), encoding="utf-8"
    )
    print(f"\n{len(figure_rows)} figures from {len(pdfs)} PDFs. Loading CLIP...", flush=True)
    classifier = aims_parser.ImageClassifier(reference_dir=str(AIMS / "reference_images"))
    kept = 0
    scored_path = OUT / "scored.jsonl"
    with scored_path.open("w", encoding="utf-8") as fh:
        batch = 8
        for start in range(0, len(figure_rows), batch):
            chunk = figure_rows[start:start + batch]
            pil_images = []
            ok_idx = []
            for j, row in enumerate(chunk):
                try:
                    pil_images.append(Image.open(io.BytesIO((OUT / row["file_name"]).read_bytes())).convert("RGB"))
                    ok_idx.append(j)
                except Exception as exc:
                    out = dict(row)
                    out["decision"] = False
                    out["reason"] = f"unreadable: {exc}"
                    fh.write(json.dumps(out) + "\n")
            embeddings = classifier.encode_batch(pil_images) if pil_images else None
            for local_i, j in enumerate(ok_idx):
                result = classifier.classify_image(pil_images[local_i], embedding=embeddings[local_i])
                out = dict(chunk[j])
                out.update({
                    "decision": bool(result["decision"]),
                    "keep_score": result["keep_score"],
                    "reject_score": result["reject_score"],
                    "keep_label": result["keep_label"],
                    "reject_label": result["reject_label"],
                    "reason": result["reason"],
                })
                kept += int(out["decision"])
                fh.write(json.dumps(out) + "\n")
            print(f"  scored {min(start + batch, len(figure_rows))}/{len(figure_rows)}  kept {kept}", flush=True)
    print(f"CLIP kept {kept} / {len(figure_rows)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
