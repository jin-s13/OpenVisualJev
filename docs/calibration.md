# Temperature calibration

The ordinary predictor returns raw candidate probabilities. A positive scalar
temperature rescales logits and preserves their argmax. The existing calibration
artifact binds the temperature to a bundle hash, model identity, and four-candidate
calibration protocol.

```python
from open_visualjev.calibrated_predictor import CalibratedVisualChoicePredictor

model = CalibratedVisualChoicePredictor(
    "/path/to/base", "/path/to/bundle", "/path/to/temperature.json"
)
result = model.predict("/path/to/image.jpg", questions, dataset="pets")
```

For the unified six-dataset bundle, provide the dataset identity: `pets`, `cub`,
`cars`, `dogs`, `flowers`, or `aircraft`. Legacy VQA bundles use the full-image
protocol and omit `dataset`. The wrapper checks these protocols explicitly.

Fit temperatures using a separate calibration split and freeze them before test
evaluation. The compact package consumes existing audited sidecars; release URLs
and a general-purpose fitting/export workflow remain pending. A newly trained
bundle initially supports raw probabilities. Model-specific temperatures from
the paper belong to their original model bundles.

The training-time mean-confidence penalty evaluated in the report is a separate
loss ablation. The compact trainer currently implements candidate cross-entropy.
