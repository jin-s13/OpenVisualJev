# Models, datasets and licensing

- Base model: [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct).
  Download the original model separately and retain its model card and license.
- Model bundles: LoRA adapter, pointer `readout.pt`, and a SHA-256-bound
  `bundle.json`. Public download URLs are pending. The training tool creates this
  structure for new user runs.
- Temperature sidecars: specific to a bundle, split and candidate-count protocol.
- Data: obtain benchmark images and annotations from their original providers;
  their individual terms apply. Original benchmark images and annotation archives are not included here.

The author is Sheng Jin, affiliated with SenseTime Research and Tetras.AI.
Project code is licensed under the [MIT License](../LICENSE), copyright 2026
Sheng Jin. Third-party templates, dependencies, datasets and pretrained models
retain their respective licenses. Checkpoint hosting is pending; source code is maintained at [jin-s13/OpenVisualJev](https://github.com/jin-s13/OpenVisualJev).
Original benchmark images and pretrained weights are obtained separately.

The architecture
uses PyTorch Transformer blocks and the Transformers/PEFT implementation of Qwen.
Dependency licenses remain their respective upstream licenses.
