# Initial visual curation

This is an initial visual-screen example of the dataset policy, not a claim that the
10-image sample is a representative training set. `curation_review.jsonl`
records a keep/drop decision and reason for every sample. `metadata_curated.jsonl`
contains the four provisional keepers in imagefolder-compatible JSONL form.

The screen keeps clear, interpretable nuclear engineering schematics alongside
photographs as a separately identifiable style category. It excludes plots,
contour/map visualizations, process-flow diagrams, and extraction defects.
`sample_06.jpeg` is a borderline facility schematic, so it is retained only in
the schematic track and should be easy to exclude from a photo-only run.

All four captions were rewritten from the source captions and visible image
content. This is an AI-assisted first pass, not a nuclear-engineering expert
review. Do not use this tiny sample as a final model-training set.

For a large directory, `review_images.py` builds a paginated browser review queue
without sending images to a VLM. After labeling, export `review-decisions.jsonl`
and apply it to the original metadata to create the curated imagefolder manifest.
