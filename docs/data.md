# Data contract

Requests and labels are separate UTF-8 JSONL files. Blank lines, duplicate JSON
keys, nonfinite numbers, missing images, and content-hash mismatches are errors.
Question IDs are unique across the file. Images are paths relative to `--images`.

Request (one line in the file):

```json
{"image":"example.jpg","image_sha256":"<actual lowercase SHA-256>","questions":[{"id":"example-1","question":"Which animal is shown?","candidates":[{"id":"cat","text":"a cat"},{"id":"dog","text":"a dog"}]}]}
```

Reference:

```json
{"question_id":"example-1","candidates":[{"id":"cat","text":"a cat"},{"id":"dog","text":"a dog"}],"correct_id":"cat"}
```

Replace the illustrative hash with the image's actual digest. Request and
reference candidates must match exactly, including IDs, text, and order.
Training requires nonblank, unique candidate text; inference preserves literal
blank or duplicate text to support the existing VQA event contract. The scorer
aggregates duplicate literal-text probability and reports that event's accuracy,
NLL and summed Brier, along with designated-slot accuracy. Its ECE uses fifteen
equal-width bins; the paper's recognition tables use equal-count bins.

Keep training, development, calibration, and test images disjoint. Deduplicate
before splitting. A manifest can contain varying candidate counts; the compact
trainer consumes those candidate sets as given.

The CLI loads full images. For the paper's Aircraft protocol, prepare separate
images with the bottom twenty pixels removed and record their hashes, or use
`VisualChoicePredictor.predict(..., crop_bottom_pixels=20)` through the Python API.
Preserve the original files. The calibrated unified-bundle API accepts
`dataset="aircraft"` and applies its bound crop protocol.
