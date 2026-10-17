<h1 align="center">OpenVisualJev</h1>
<h3 align="center">Design and Analysis of a Multimodal Decision Model</h3>
<p align="center"><strong>Sheng Jin</strong><br>SenseTime Research and Tetras.AI</p>
<p align="center">
  <a href="paper/open-visualjev.pdf">Paper</a> ·
  <a href="https://github.com/jin-s13/ai-research-writing-skill">Research & Writing Skill</a> ·
  <a href="README.zh-CN.md">中文</a>
</p>

## Overview

**OpenVisualJev** is a multimodal decision model that directly assigns probabilities to runtime-defined choices. Built on **Qwen3-VL-4B**, it combines visual–language representations with a shared dynamic pointer for image recognition and visual question answering.

- **Dynamic choices:** natural-language candidates define the decision space for each question.
- **Set-aware scoring:** isolated candidate encoding and position-free set interaction support joint comparison and permutation-equivariant scores.
- **Probability calibration:** calibration-aware training and held-out temperature scaling refine the output probabilities.

![OpenVisualJev architecture: shared context, isolated branches, set interaction and dynamic pointer](paper/figures/set-pointer-method.png)

The figure illustrates tree-mask execution. The paper's main results use ordinary batched branches; tree-mask inference is an [explicit option](docs/tree-inference.md).

## A paper produced with ai-research-writing-skill

**The entire paper is an output of the [ai-research-writing-skill](https://github.com/jin-s13/ai-research-writing-skill) research and writing workflow.** Under Sheng Jin's direction, AI assistance supported literature review, method and experiment design, implementation, experiment execution and analysis, figure preparation, manuscript drafting, and revision.

Experimental results come from actual model runs and recorded evaluation scripts. The author guided the research, reviewed the evidence and manuscript, and takes responsibility for the final work. See the paper's **AI usage disclosure** for the full account.

## Results

Qwen3-VL-4B, matched complete-method training, one recognition training seed:

| Evaluation | Complete-candidate LM | Set Pointer |
|---|---:|---:|
| Six-dataset recognition accuracy | 90.09% | **90.35%** |
| Questions changing answer under candidate reordering ↓ | 108 / 755 | **0 / 755** |
| Recognition with 20 difficult candidates | 80.26% | **82.06%** |

See the [paper](paper/open-visualjev.pdf) for relation tasks, natural VQA, calibration, and component analyses.

## Quick start

The tested environment uses Python 3.10, PyTorch 2.8.0, Transformers 4.57.3, and PEFT 0.17.1 on CUDA.

```bash
git clone https://github.com/jin-s13/OpenVisualJev.git
cd OpenVisualJev
python -m venv .venv
source .venv/bin/activate
pip install -e '.[vlm]'
```

Download the [Qwen3-VL-4B-Instruct base model](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) separately. Inference also requires a trained adapter/head bundle; **checkpoint downloads are pending**. See [model assets](docs/assets.md) for preparation and [training](docs/training.md) to create a bundle.

```python
from open_visualjev.predictor import VisualChoicePredictor

model = VisualChoicePredictor(
    "/path/to/Qwen3-VL-4B-Instruct",
    "/path/to/checkpoint-bundle",
)
answers = model.predict("/path/to/image.jpg", [{
    "id": "animal",
    "question": "Which animal is shown?",
    "candidates": [
        {"id": "cat", "text": "a cat"},
        {"id": "dog", "text": "a dog"},
        {"id": "bird", "text": "a bird"},
    ],
}])
print(answers)
```

The interface accepts an image and one or more questions, each with 2–20 candidates, and returns candidate IDs, logits, and probabilities. For calibrated outputs, use the [calibrated predictor](docs/calibration.md).

## Training and evaluation

| Task | Guide |
|---|---|
| Prepare images and candidate questions | [Data](docs/data.md) |
| Train on custom data | [Training](docs/training.md) |
| Evaluate predictions and probability quality | [Evaluation](docs/evaluation.md) |

## Citation

```bibtex
@techreport{jin2026openvisualjev,
  title       = {{OpenVisualJev}: Design and Analysis of a Multimodal Decision Model},
  author      = {Jin, Sheng},
  institution = {SenseTime Research and Tetras.AI},
  year        = {2026},
  type        = {Technical Report},
  url         = {https://github.com/jin-s13/OpenVisualJev}
}
```

## Acknowledgements and license

This project builds on [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL), PyTorch, Transformers, and PEFT. Its research and writing workflow uses [ai-research-writing-skill](https://github.com/jin-s13/ai-research-writing-skill).

Code is released under the [MIT License](LICENSE). Pretrained models, datasets, and third-party assets retain their respective licenses.
