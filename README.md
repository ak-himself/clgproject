# Waste Classification Lab

A reproducible TrashNet waste-image classification project comparing
ImageNet-pretrained MobileNetV2 and EfficientNetB0.

The project evaluates on a held-out test split and provides a Gradio GUI for
uploading individual or multiple images.

## Why transfer learning?

The Keras MobileNetV2 and EfficientNetB0 backbones use weights pretrained on
ImageNet. We replace their original 1,000-class classifier heads with a
TrashNet-specific head and fine-tune them. We do not train either backbone from
scratch.

## Dataset

The [original TrashNet dataset](https://github.com/garythung/trashnet) contains
2,527 images across cardboard, glass, metal, paper, plastic, and trash. Most
images use a plain posterboard background, so results should not be treated as
evidence of robust real-world performance.

Expected local path:

```text
data/raw/dataset-resized/
├── cardboard/
├── glass/
├── metal/
├── paper/
├── plastic/
└── trash/
```

The dataset archive and raw image folders are local-only and ignored by Git.

## Environment

Use Python 3.12.8 via pyenv on CachyOS. TensorFlow will use the CPU on systems
without a supported NVIDIA CUDA GPU.

```fish
cd /home/ak/Code/clgproject
pyenv shell 3.12.8
python -m pip install -r requirements.txt
```

A dedicated virtual environment is preferable for repeatability, but avoid
installing project packages into system Python.

## Prepare dataset splits

Extract the TrashNet archive so each class has its own folder, then run:

```fish
python src/prepare_data.py
```

The script creates reproducible stratified splits of approximately 70% training,
15% validation and 15% test in `data/processed/`. The existing manifests use
the same class order and paths expected by the trained checkpoints.

## Train and evaluate

Existing TrashNet checkpoints and reports are kept under their original names.
Do not retrain them unless intentionally replacing those artifacts.

```fish
python src/train.py --model mobilenetv2 --head-epochs 5 --fine-tune-epochs 3
python src/evaluate.py --model mobilenetv2
```

To use EfficientNetB0, replace `mobilenetv2` with `efficientnetb0`. Training
includes mild image augmentation and class weighting. It uses CPU-oriented
thread limits to keep a laptop usable.

Model checkpoints and metadata are saved under `models/`. Evaluation metrics
and confusion matrices are saved under `reports/`.

## Image-classification GUI

Install the dependencies, then launch the local Gradio application from the
repository root:

```fish
cd /home/ak/Code/clgproject
pyenv shell 3.12.8
python -m pip install -r requirements.txt
python src/app.py
```

Open the local URL Gradio prints. The GUI defaults to the TrashNet EfficientNetB0
checkpoint and supports:

- one uploaded image with its predicted class, top-three model scores and
  inference latency;
- multiple uploaded images with a results table; and
- comparing MobileNetV2 and EfficientNetB0 on the same image or batch.

Models are loaded once when the application starts. If a checkpoint or its
metadata is missing or cannot be loaded, the GUI reports the problem and keeps
the other model usable.

The classifier is a closed-set model: every input is assigned one of its known
classes. It may confidently misclassify unrelated or ambiguous photos, so score
values should not be interpreted as calibrated certainty. This interface
classifies one image as a whole; it does not locate multiple objects.

## Evaluation protocol

- Use validation results for model selection.
- Use the held-out test set only for final metrics.
- Report accuracy, macro F1, per-class precision/recall and a confusion matrix.
- Maintain an additional external challenge set of independently sourced images
  that are never used for training or tuning.
- Track source URLs, labels, license/usage terms and potential duplicates.
  Never report training images as unseen test examples.

## Limitations and next experiments

TrashNet is small and uses relatively plain backgrounds. It does not guarantee
reliable classification of every external image or every object in a cluttered
scene.

A useful next experiment is to analyze class-wise errors and separately
evaluate an external challenge set. If the intended task changes to finding
several pieces of litter in a scene, use an object-detection dataset such as
[TACO](https://github.com/pedrocampos/taco) rather than treating the entire
scene as one classification sample.

## Data acknowledgement

Cite TrashNet in any report and review the source dataset's reuse terms before
redistributing images. This repository does not bundle the full dataset.
