# How the pipeline works (plain-English walkthrough)

This explains, from scratch, **how the code turns a search query into a folder of
captioned nuclear images.** No prior knowledge assumed. The most important thing
to understand first:

> **Nothing here is AI-generated. No images are "created."** The code finds real
> figures that already exist inside published PDF documents (DOE reports and
> patents), cuts each one out, and grabs the caption the author already wrote for
> it. Think "automated, targeted screenshot-and-copy," not "image generator."

There are two harvesters (`harvest_osti.py`, `harvest_patents.py`) and one
post-processing stage (`recaption.py`). They all end in the same output format.

---

## The mental model

A scientific PDF is really two things interleaved: **pictures** (figures) and
**text**. Every figure has a caption — a line of text like *"Figure 3. Reactor
core cross-section."* A human reading the PDF pairs them effortlessly: the caption
sits right under the picture.

Our job is to make a program do that pairing automatically, thousands of times,
across thousands of PDFs. Once we can reliably say *"this picture goes with this
caption,"* we have an (image, caption) training pair — exactly what a text-to-image
model needs.

Everything below is in service of that one task: **find pictures, find captions,
pair them correctly.**

---

## `harvest_osti.py` — step by step

### Step 1: Search (find documents, not images yet)

OSTI.gov (the DOE report repository) has a public **API** — a URL you can request
that returns data instead of a web page. We ask it:

```
https://www.osti.gov/api/v1/records?q=PWR+fuel+assembly&rows=10
```

It replies with a list of **reports** in JSON (structured data). For each report
we get its `osti_id` (a unique number) and a link to its full-text PDF. At this
point we have a *list of documents*, no images yet.

```python
def search_osti(query, rows):
    r = requests.get(OSTI_SEARCH, params={"q": query, "rows": rows})
    return r.json()          # list of report records
```

### Step 2: Download the PDF

For each report, we fetch its PDF and save it to disk. We sanity-check that the
downloaded bytes actually start with `%PDF` (the signature every PDF begins with),
so we don't accidentally save an error page.

```python
if r.content[:4] != b"%PDF":   # not a real PDF -> skip
    return None
```

### Step 3: Extract figures and pair captions (the core trick)

We open the PDF with **PyMuPDF** (a library that can read a PDF's internal
structure — every image and every piece of text, *with the x/y coordinates of
where each sits on the page*). Those coordinates are the key.

For each page we build two lists:

**(a) The captions.** We look at every text block and keep only the ones that
*start with* "Figure N" or "Fig. N". But there's a trap: the body text also says
things like *"Figure 9 plots the net mass..."* — that's a **reference to** a
figure, not a caption. We filter those out with a rule: if "Figure N" is
immediately followed by a verb like *shows / plots / compares / illustrates*,
it's a reference (a sentence), not a caption (a title). Skip it.

```python
CAP_RE     = r"^\s*fig(?:ure)?\.?\s*\d+"          # starts with "Figure N"
INTEXT_RE  = r"^fig...\s*\d+\s+(shows|plots|compares|...)"  # a sentence -> reject
```

**(b) The images.** We get every embedded image on the page **and its bounding
box** (the rectangle where it's placed). We throw away anything tiny (< 120px) —
those are logos, header rules, math symbols, not real figures.

**Now the pairing.** Each caption has a vertical position (its y-coordinate). Each
image has one too. We match each caption to the **image whose box is vertically
closest to it** — because a caption physically sits right above or below its
figure. That proximity is how we know *"Figure 2-1" the text belongs to this
particular picture and not another one on the page.*

```python
best = min(images, key=lambda img: abs(img.y - caption.y))   # nearest image
```

That's the whole idea. On a page with one figure and one caption it's trivially
correct; on a page with several, the nearest-neighbor rule usually gets it right
(and mis-pairs are the main known weakness — see caveats).

### Step 4: Write the dataset

For each pair we save the image as a file and append one line to
`metadata.jsonl`:

```json
{"file_name": "images/osti_3384931_000.jpeg", "text": "Fig. 1. Reference radial layout of the HPMR configuration..."}
```

A `.jsonl` file is just "one JSON object per line." This exact layout — an
`images/` folder plus a `metadata.jsonl` — is the **HuggingFace imagefolder
format**, which the paper's fine-tuning code reads directly:

```python
load_dataset("imagefolder", data_dir="output")
```

So the output drops straight into their existing training pipeline. No conversion.

---

## `harvest_patents.py` — same goal, different document shape

Patents also have figures + captions, but arranged differently, so the pairing
logic changes.

**Discovery** uses Google Patents' keyless `xhr/query` endpoint (returns JSON like
an API). Crucially we add a **CPC class filter** `cpc=G21C` ("Nuclear reactors").
Without it, a keyword search for "control rod drive" returns *surgical-robot*
patents (they mention "rod" and "drive") — 50 irrelevant figures each. The class
filter keeps us in the nuclear domain.

**The structural difference:** in a patent, the drawings live on their own pages
(one big drawing per sheet, each labeled "FIG. 1"), while the *descriptions* of
those drawings live in a separate "Brief Description of the Drawings" section in
the body text. They are **not** next to each other, so the OSTI nearest-neighbor
trick doesn't apply. Instead:

1. **Parse the descriptions** from the body text into a lookup table:
   `{"1": "FIG. 1 is a schematic cross-sectional view of a control drum...", ...}`
2. **Identify drawing pages** (little text, a "FIG. N" label, one big image).
3. For each drawing page, read which figure number is printed on it, pull the
   sheet image, and attach the matching description as the caption.
4. If no description parses cleanly, fall back to a caption built from the patent
   title (`FIG. 3: engineering drawing from patent "Mixed oxide fuel assembly"`).

Patent captions are often *richer* than report captions (patent law requires every
drawing be described), which is their appeal. The trade-off: patent drawings are
black-and-white line art with reference numerals — a narrow visual style (caveat).

---

## `recaption.py` — the quality stage (post-processing)

The harvesters keep **every** figure, including data plots/charts we don't want.
`recaption.py` fixes that using a **vision-language model** (a model that can look
at an image and answer questions about it). For each harvested figure it asks the
model two things:

1. **Keep or drop?** "Is this a real nuclear object (schematic/photo/cutaway) or a
   plot/chart/table?" → drops the graphs and tangential photos.
2. **Better caption.** "Describe what's actually shown." → replaces the terse
   source caption with a rich one (fixing the paper's "captions too short" gap).

It has three **backends** it auto-detects:
- **ollama** — a local model server; runs free on your Ryzen box. *(recommended)*
- **anthropic** — a cloud API (needs a key, costs money).
- **passthrough** — if neither is available (e.g. this low-power machine), it just
  copies rows through unchanged so nothing crashes. No filtering happens in this
  mode — it's only a plumbing check.

It writes results after every row, so a slow local run is **resumable** (stop and
restart without losing progress).

---

## The data's journey, end to end

```
 "PWR fuel assembly"          <- your query
        │  search_osti / Google Patents xhr
        ▼
 list of reports/patents      <- documents, no images yet
        │  download_pdf
        ▼
 PDF files on disk
        │  extract_figures  (find images + captions, pair by position/figure-number)
        ▼
 output/images/*.png  +  metadata.jsonl     <- RAW (includes plots)
        │  recaption.py  (VLM: drop plots, enrich captions)   ← run on Ryzen box
        ▼
 metadata_filtered.jsonl      <- CLEAN, training-ready (image, caption) pairs
        │  load_dataset("imagefolder", ...)
        ▼
 fine-tuning (the paper's existing pipeline)
```

---

## Why the pairing is reliable enough (and where it isn't)

- **Reliable:** single-figure pages (the common case), because "nearest caption"
  is unambiguous. Verified by eye on sampled outputs — real labeled cross-sections
  and layouts landed with their correct captions.
- **Weak spots (documented in the README caveats):**
  - Multi-panel pages can mis-pair (two figures, two captions, wrong assignment).
  - A few in-text "Figure N ..." references slip past the verb filter.
  - No cross-document dedup yet (the same figure can appear in related reports).
  - Patent description-parsing sometimes misses, falling back to the title caption.

None of these break the approach; they set the "spot-check before training" bar,
which is exactly what the `recaption.py` VLM stage plus a human glance handle.
