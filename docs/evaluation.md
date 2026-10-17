# Evaluation

Run commands from the repository root. Obtain the matching base model and trained bundle as described in [assets](assets.md).

```sh
CUDA_VISIBLE_DEVICES=0 python tools/run_choice_requests.py \
  --base /path/to/Qwen3-VL-4B-Instruct --bundle /path/to/bundle \
  --requests data/test.requests.jsonl --images data/images --output outputs/test
python tools/evaluate_choice_predictions.py \
  --references data/test.references.jsonl \
  --predictions outputs/test/predictions.jsonl --output outputs/metrics.json
```

Predictions retain question and candidate IDs. Labels stay in a separate reference file. Use the [data contract](data.md) to construct matching inputs; invalid or mismatched records are errors. Output directories must be new.

The compact predictor uses the validated ordinary BF16 inference path. FP16, compilation and shared-prefix optimization probes remain in the research workspace; their speed measurements are not guarantees for this API.
