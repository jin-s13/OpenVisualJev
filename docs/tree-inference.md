# Tree-mask inference

`VisualChoicePredictor.predict` and `CalibratedVisualChoicePredictor.predict`
accept `execution="tree"` and `max_tree_tokens=4096`. The default remains
`execution="ordinary"` while numerical validation is reviewed.

```python
results = predictor.predict(image_path, questions, execution="tree")
# Unified calibrated bundles also require their dataset:
results = calibrated.predict(image_path, questions, dataset="pets", execution="tree")
```

A request packs one image prefix, a prefix for each question, and isolated
query/candidate suffixes into a single backbone forward. Tokens attend to their
own causal predecessors and ancestors. The original branch MRoPE coordinates
are retained. Candidate representations feed the unchanged question-local Set
module and pointer. Shared projection is applied independently to each candidate.

This implementation uses a dense four-dimensional SDPA mask. The mask grows
quadratically with packed length; exceeding `max_tree_tokens` raises an error.
There is no truncation or automatic switch to another execution mode. Image
preprocessing and the visual backbone run once; preprocessing output tensors
are repeated temporarily to validate the original branch input contract.

Ordinary execution batches a query and its candidates in one forward per
question and repeats image encoding across batch rows. Tree execution changes
matrix shapes and floating-point reductions. Mathematical visibility equivalence
does not imply bitwise equality with ordinary execution. The original BF16
path remains the reference for the paper results. Temperature scaling preserves
argmax within an execution path but does not remove numerical differences
between execution paths. No tree speedup or unchanged calibration is claimed
from the interface alone.

## Bounded validation

On six fixed real images and 36 question executions including invariance probes, FP32/math produced no decision changes against ordinary execution; maximum probability difference was 3.40e-5. Stable FP16 also produced no changes, with maximum difference 0.01324. Native BF16 changed 5 of 36 decisions in the original implementation. None of these configurations achieved bitwise equality. These are engineering checks, not full-test-set quality estimates. Forty-eight related regression tests passed.
