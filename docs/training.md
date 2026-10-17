# Training and evaluation

Prepare separate request and reference JSONL files using the
[data contract](data.md). Training supports Set and Independent heads,
LoRA adaptation, and candidate counts supplied in the input manifests.

```bash
# Validate images, candidate identities, and labels before loading the VLM.
python tools/train_choices.py --requests data/train.requests.jsonl \
  --references data/train.references.jsonl --images data/images \
  --config configs/set.json --output outputs/preflight --validate-only

CUDA_VISIBLE_DEVICES=0 python tools/train_choices.py \
  --base /path/to/Qwen3-VL-4B-Instruct \
  --requests data/train.requests.jsonl --references data/train.references.jsonl \
  --images data/images --config configs/set.json --output outputs/set

CUDA_VISIBLE_DEVICES=0 python tools/run_choice_requests.py \
  --base /path/to/Qwen3-VL-4B-Instruct --bundle outputs/set/bundle \
  --requests data/test.requests.jsonl --images data/images --output outputs/test

python tools/evaluate_choice_predictions.py \
  --references data/test.references.jsonl \
  --predictions outputs/test/predictions.jsonl --output outputs/metrics.json
```

Use `configs/independent.json` for the independent head. The native-order direct
LM baseline is `tools/run_lm_choice_requests.py`; `--adapter` is optional.
Every command refuses to overwrite existing results.

The trainer uses cross-entropy on user-supplied candidate sets and exports the final epoch. Supply separate training, validation and test data for your application.
