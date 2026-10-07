#!/usr/bin/env python3
"""Build a zero-VLM browser review queue and apply its decisions to metadata."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from urllib.parse import quote

CAPTION_JUNK_RE = re.compile(
    r"\b(?:bar\s+charts?|histograms?|contour\s+(?:plots?|maps?)|(?:2|3)[- ]?d\s+plots?|"
    r"scatter\s+plots?|flow\s*charts?|data\s+flows?|flow\s+diagrams?|"
    r"map\s+layouts?|maps?\s+of\s+(?:values|data|inventory)|"
    r"(?:line|area|surface)\s+plot|(?:temperature|burnup|pressure|power)\s+"
    r"(?:versus|vs\.?|as\s+a\s+function\s+of)|"
    r"plots?|graphs?|spectra|spectrum|versus|vs\.?|as a function of|"
    r"histograms?|micrographs?|uncertainty|dose rate|reactivity swing)\b",
    re.IGNORECASE,
)
CAPTION_DIAGRAM_RE = re.compile(
    r"\b(?:cross[- ]sections?|schematics?|diagrams?|layouts?|cutaways?|"
    r"\bcad\b|geometr(?:y|ies)|arrangements?|lattices?|assembl(?:y|ies)|"
    r"configurations?|horizontal cut|vertical cut|core model)\b",
    re.IGNORECASE,
)


def read_records(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                records.append(json.loads(line))
    return records


def build_review(records: list[dict], root: Path, output: Path) -> int:
    root = root.resolve()
    output = output.resolve()
    items = []
    missing = duplicate = unsafe = exact_duplicates = 0
    seen: set[str] = set()
    seen_hashes: dict[str, str] = {}
    for record in records:
        name = str(record.get("file_name") or "")
        if not name or name in seen:
            duplicate += bool(name)
            continue
        seen.add(name)
        image = (root / name).resolve()
        try:
            image.relative_to(root)
        except ValueError:
            unsafe += 1
            continue
        if not image.is_file():
            missing += 1
            continue
        digest = hashlib.sha256()
        with image.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        duplicate_of = seen_hashes.get(digest.hexdigest(), "")
        if duplicate_of:
            exact_duplicates += 1
        else:
            seen_hashes[digest.hexdigest()] = name
        try:
            rel_image = image.relative_to(output.parent).as_posix()
        except ValueError:
            rel_image = Path(os.path.relpath(image, output.parent)).as_posix()
        caption = str(record.get("text") or "")
        junk = bool(CAPTION_JUNK_RE.search(caption))
        diagram = bool(CAPTION_DIAGRAM_RE.search(caption)) and not junk and not duplicate_of
        items.append({
            "file_name": name,
            "image_url": quote(rel_image, safe="/:"),
            "caption": caption,
            "source": record.get("source", ""),
            "source_id": record.get("osti_id") or record.get("patent_number") or "",
            "auto_drop": junk or bool(duplicate_of),
            "auto_candidate": diagram,
            "auto_reason": ("exact_duplicate_of:" + duplicate_of if duplicate_of else
                            "caption_prefilter" if junk else ""),
            "duplicate_of": duplicate_of,
        })
    items.sort(key=lambda x: (not x["auto_candidate"], x["auto_drop"], x["file_name"]))

    data = json.dumps(items, ensure_ascii=False).replace("<", "\\u003c")
    page = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Nuclear image review</title>
<style>
body{font:16px system-ui,sans-serif;margin:0 auto;max-width:1100px;padding:0 1rem 2rem;background:#f6f7f8;color:#18212b}
header{position:sticky;top:0;background:#f6f7f8;padding:.7rem 0;border-bottom:1px solid #ccd2d8;z-index:2}
.toolbar{display:flex;gap:.6rem;align-items:center;flex-wrap:wrap}.toolbar button,.toolbar select{padding:.45rem .7rem}
#focus{display:grid;grid-template-columns:1fr;gap:.6rem;margin-top:.8rem}
#focus img{width:100%;max-height:68vh;object-fit:contain;background:#fff;border:2px solid #d5dbe1;border-radius:8px}
#focus.schematic img,#focus.photo img{border-color:#28905a}#focus.drop img{border-color:#c54444}
.caption{font-size:1rem;line-height:1.35;margin:.2rem 0}.keys{color:#5c6975;font-size:.9rem}
#cards{display:none;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:1rem;margin:1rem 0}
body.grid #cards{display:grid}body.grid #focus{display:none}
article{background:white;border:2px solid #d5dbe1;border-radius:8px;padding:.7rem;min-width:0}
article[data-vote="photo"],article[data-vote="schematic"]{border-color:#28905a}article[data-vote="drop"]{border-color:#c54444}
article img{width:100%;height:210px;object-fit:contain;background:#e8ebee}h3{font-size:.9rem;overflow-wrap:anywhere;margin:.45rem 0}
.cardcap{font-size:.83rem;max-height:5.5em;overflow:auto;color:#3e4b58}.buttons{display:flex;gap:.35rem;margin-top:.6rem;flex-wrap:wrap}
.buttons button{flex:1;padding:.45rem .2rem}button.active{background:#243b53;color:white}small{color:#5c6975}
</style>
<header><div class="toolbar"><strong>Fast review</strong><span id="status"></span>
<label>Queue <select id="queue"><option value="candidates" selected>Likely diagrams</option><option value="unlabeled">Not yet labeled</option><option value="all">All images</option><option value="drops">Auto-rejected plots</option></select></label>
<button id="prev">Previous</button><span id="page"></span><button id="next">Next</button>
<button id="mode">Grid</button>
<label id="sizewrap">Per page <select id="size"><option>30</option><option selected>50</option><option>100</option></select></label>
<button id="export">Download decisions</button><button id="clear">Clear votes</button></div>
<p class="keys">Keys: <b>S</b> keep schematic, <b>P</b> keep photo, <b>D</b> drop, <b>U</b> unsure, <b>←</b> <b>→</b> move. A vote advances to the next image. Likely diagrams are first; plot captions are pre-marked drop.</p></header>
<section id="focus"></section>
<main id="cards"></main>
<script>
const items=__ITEMS__, key="nuclear-image-review-v2:"+location.pathname;
let votes={};try{votes=JSON.parse(localStorage.getItem(key)||"{}")}catch{}
let page=0, index=0, grid=false;
const el=id=>document.getElementById(id);
function queue(){
 const mode=el("queue").value;
 return items.filter(x=>{
  if(mode==="candidates") return x.auto_candidate;
  if(mode==="drops") return x.auto_drop;
  if(mode==="unlabeled") return !votes[x.file_name];
  return true;
 });
}
function voteOf(x){return (votes[x.file_name]||{}).verdict||(x.auto_drop?"drop":"");}
function setVote(x,v){
 votes[x.file_name]={verdict:v,category:v==="photo"?"component_photo":v==="schematic"?"schematic":"",reason:v==="drop"?(x.auto_reason||"visual_review"):"visual_review"};
 try{localStorage.setItem(key,JSON.stringify(votes))}catch{}
 const list=queue();
 const at=list.findIndex(y=>y.file_name===x.file_name);
 if(at>=0 && at<list.length-1) index=at+1;
 render();
}
function saveOnly(){try{localStorage.setItem(key,JSON.stringify(votes))}catch{}render()}
function render(){
 const list=queue();
 const labeled=Object.keys(votes).length;
 const cand=items.filter(x=>x.auto_candidate).length;
 el("status").textContent=`${labeled} labeled · ${cand} likely diagrams · ${items.filter(x=>x.auto_drop).length} auto-rejected plots`;
 document.body.classList.toggle("grid", grid);
 el("mode").textContent=grid?"One at a time":"Grid";
 el("sizewrap").style.display=grid?"":"none";
 if(!grid){
  if(!list.length){el("focus").innerHTML="<p>Nothing in this queue.</p>";el("page").textContent="0 / 0";return;}
  index=Math.max(0,Math.min(index,list.length-1));
  const x=list[index];
  el("page").textContent=`${index+1} / ${list.length}`;
  el("focus").className=voteOf(x);
  el("focus").innerHTML="";
  const img=document.createElement("img");img.src=x.image_url;img.alt=x.file_name;el("focus").append(img);
  const h=document.createElement("h2");h.textContent=`${index+1}. ${x.file_name}`;h.style.margin="0";el("focus").append(h);
  const cap=document.createElement("p");cap.className="caption";cap.textContent=x.caption;el("focus").append(cap);
  const row=document.createElement("div");row.className="buttons";
  for(const [v,label] of [["schematic","S schematic"],["photo","P photo"],["drop","D drop"],["unsure","U unsure"]]){
   const b=document.createElement("button");b.textContent=label;if(voteOf(x)===v)b.className="active";
   b.onclick=()=>setVote(x,v);row.append(b);
  }
  el("focus").append(row);
  return;
 }
 const size=Number(el("size").value)||50;
 const pages=Math.max(1,Math.ceil(list.length/size));
 page=Math.max(0,Math.min(page,pages-1));
 const start=page*size, batch=list.slice(start,start+size);
 el("page").textContent=`${page+1} / ${pages} (${list.length})`;
 el("cards").innerHTML="";
 for(const [j,x] of batch.entries()){
  const card=document.createElement("article");card.dataset.vote=voteOf(x);
  const img=document.createElement("img");img.loading="lazy";img.src=x.image_url;img.alt=x.file_name;card.append(img);
  const h=document.createElement("h3");h.textContent=`${start+j+1}. ${x.file_name}`;card.append(h);
  const cap=document.createElement("div");cap.className="cardcap";cap.textContent=x.caption;card.append(cap);
  const row=document.createElement("div");row.className="buttons";
  for(const [v,label] of [["schematic","Schematic"],["photo","Photo"],["drop","Drop"],["unsure","Unsure"]]){
   const b=document.createElement("button");b.textContent=label;if(voteOf(x)===v)b.className="active";
   b.onclick=()=>{votes[x.file_name]={verdict:v,category:v==="photo"?"component_photo":v==="schematic"?"schematic":"",reason:v==="drop"?(x.auto_reason||"visual_review"):"visual_review"};saveOnly()};
   row.append(b);
  }
  card.append(row);el("cards").append(card);
 }
}
function move(delta){
 const list=queue();
 if(grid){page+=delta;render();return;}
 index=Math.max(0,Math.min(list.length-1,index+delta));
 render();
}
el("prev").onclick=()=>move(-1);
el("next").onclick=()=>move(1);
el("queue").onchange=()=>{index=0;page=0;render()};
el("size").onchange=()=>{page=0;render()};
el("mode").onclick=()=>{grid=!grid;render()};
el("clear").onclick=()=>{if(confirm("Clear every saved vote?")){votes={};saveOnly()}};
el("export").onclick=()=>{
 const rows=items.map(x=>{const v=votes[x.file_name]||{verdict:x.auto_drop?"drop":"unsure",category:"",reason:x.auto_reason||"not_reviewed"};return JSON.stringify({file_name:x.file_name,...v})});
 const blob=new Blob([rows.join("\n")+"\n"],{type:"application/x-ndjson"});
 const a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download="review-decisions.jsonl";a.click();URL.revokeObjectURL(a.href);
};
document.addEventListener("keydown",ev=>{
 if(ev.target.matches("input,textarea,select")) return;
 const list=queue(); const x=list[grid?0:index];
 const k=ev.key.toLowerCase();
 if(k==="arrowright"){ev.preventDefault();move(1);}
 else if(k==="arrowleft"){ev.preventDefault();move(-1);}
 else if(!grid && x && k==="s") setVote(x,"schematic");
 else if(!grid && x && k==="p") setVote(x,"photo");
 else if(!grid && x && k==="d") setVote(x,"drop");
 else if(!grid && x && k==="u") setVote(x,"unsure");
});
render();
</script></html>'''.replace("__ITEMS__", data)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding="utf-8")
    print(f"review page: {output}")
    print(f"images: {len(items)}  likely diagrams: {sum(x['auto_candidate'] for x in items)}  "
          f"auto-rejected: {sum(x['auto_drop'] for x in items)}  "
          f"exact duplicates: {exact_duplicates}  missing: {missing}  "
          f"duplicate names: {duplicate}  unsafe paths: {unsafe}")
    return 0


def apply_decisions(records: list[dict], decisions_path: Path, output: Path) -> int:
    decisions = {}
    for row in read_records(decisions_path):
        decisions[row.get("file_name", "")] = row
    kept = dropped = uncertain = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        seen = set()
        for record in records:
            name = record.get("file_name", "")
            if name in seen:
                continue
            seen.add(name)
            decision = decisions.get(name, {})
            verdict = decision.get("verdict")
            if verdict in ("photo", "schematic", "keep"):
                record["category"] = decision.get("category") or (
                    "component_photo" if verdict == "photo" else "schematic")
                record["curation_status"] = "visual_review"
                if decision.get("caption"):
                    record["text"] = decision["caption"]
                    record["caption_source"] = "visual_review"
                if decision.get("reason"):
                    record["curation_reason"] = decision["reason"]
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1
            elif verdict == "drop":
                dropped += 1
            else:
                uncertain += 1
    print(f"curated metadata: {output}  kept: {kept}  dropped: {dropped}  unresolved: {uncertain}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Review thousands of images without VLM calls")
    parser.add_argument("--in", dest="inp", type=Path, required=True, help="source metadata.jsonl")
    parser.add_argument("--root", type=Path, default=None, help="image root (default: input file parent)")
    parser.add_argument("--out", type=Path, required=True, help="review HTML or curated JSONL output")
    parser.add_argument("--apply", type=Path, help="apply decisions JSONL exported by the review page")
    args = parser.parse_args()
    records = read_records(args.inp)
    if args.apply:
        return apply_decisions(records, args.apply, args.out)
    return build_review(records, args.root or args.inp.parent, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
