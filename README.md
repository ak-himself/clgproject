# Waste Classification Lab

A reproducible waste-image classification project comparing ImageNet-pretrained MobileNetV2 and EfficientNetB0 on two datasets:
- **TrashNet:** small, controlled-background baseline.
- **RealWaste:** images captured in an authentic landfill environment.

The project evaluates on held-out test splits and provides a Gradio GUI for uploading individual images.

## Why transfer learning?

The Keras MobileNetV2 and EfficientNetB0 backbones use weights pretrained on ImageNet. We replace their original 1,000-class classifier heads with task-specific heads and fine-tune them on each waste dataset. We do not train either backbone from scratch.

## Datasets

### TrashNet (baseline)

The [original TrashNet dataset](https://github.com/garythung/trashnet) contains 2,527 images across cardboard, glass, metal, paper, plastic, and trash. Most images use a plain posterboard background, so results should not be treated as evidence of robust real-world performance.

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

### RealWaste (realistic primary dataset)

[RealWaste on the UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/908/realwaste) provides 4,752 images captured at the point of reception in a landfill. Its nine material categories are cardboard, food organics, glass, metal, miscellaneous trash, paper, plastic, textile trash, and vegetation.

Download the UCI archive from its dataset page, then extract it under `data/raw/realwaste/`. The preparation script searches nested folders to locate the class directories. Do not upload the dataset or model checkpoints to Git.

## Environment

Use Python 3.12.8 via pyenv on CachyOS. TensorFlow will use the CPU on systems without a supported NVIDIA CUDA GPU.

```fish
cd /home/ak/Code/clgproject
pyenv shell 3.12.8
python -m pip install -r requirements.txt
```

A dedicated virtual environment is preferable for repeatability, but avoid installing project packages into system Python.

## Prepare dataset splits

TrashNet (already prepared in the baseline workflow):

```fish
python src/prepare_data.py --dataset trashnet
```

RealWaste after extraction:

```fish
python src/prepare_data.py --dataset realwaste
```

Each dataset receives a reproducible stratified split of approximately 70% training, 15% validation and 15% test. TrashNet manifests remain in `data/processed/` for compatibility with the original baseline. RealWaste manifests go into `data/processed/realwaste/`.

## Train and evaluate

Existing TrashNet checkpoints/results are kept under their original names. New RealWaste checkpoints use a `realwaste_` prefix so the experiments cannot overwrite one another.

```fish
# TrashNet baseline
python src/train.py --dataset trashnet --model mobilenetv2 --head-epochs 5 --fine-tune-epochs 3
python src/evaluate.py --dataset trashnet --model mobilenetv2

# RealWaste experiment
python src/train.py --dataset realwaste --model mobilenetv2 --head-epochs 5 --fine-tune-epochs 3
python src/evaluate.py --dataset realwaste --model mobilenetv2
```

To train EfficientNetB0, replace `mobilenetv2` with `efficientnetb0`. Training includes mild image augmentation and class weighting. It uses CPU-oriented thread limits to keep a laptop usable; full CPU saturation is neither guaranteed nor required.

Model checkpoints and metadata are saved under `models/`. Evaluation metrics and confusion matrices are saved under `reports/`. These generated artifacts are ignored by Git.

## Image-classification GUI

Install the dependencies, then launch the local Gradio application from the repository root:

```fish
cd /home/ak/Code/clgproject
pyenv shell 3.12.8
python -m pip install -r requirements.txt
python src/app.py
```

Open the local URL Gradio prints. The GUI defaults to the RealWaste EfficientNetB0
checkpoint and supports:

- one uploaded image with its predicted class, top-three model scores and inference latency;
- multiple uploaded images with a results table;
- comparing MobileNetV2 and EfficientNetB0 on the same image or batch; and
- switching to the existing TrashNet checkpoints without changing the training or evaluation workflows.

Models are loaded once when the application starts. If a checkpoint or its metadata
is missing or cannot be loaded, the GUI reports the problem and keeps other available
models usable.

The classifier is a closed-set model: every input is assigned one of its known classes. It may confidently misclassify unrelated or ambiguous photos, so score values should not be interpreted as calibrated certainty. This interface classifies one image as a whole; it does not locate multiple objects.

## Evaluation protocol

- Use validation results for model selection.
- Use the held-out test set only for the final dataset-specific metrics.
- Report accuracy, macro F1, per-class precision/recall and a confusion matrix.
- Maintain an additional external challenge set of independently sourced images that are never used for training or tuning.
- Track source URLs, labels, license/usage terms and potential duplicates. Never report training images as unseen test examples.

## Limitations and next experiments

TrashNet is small and uses relatively plain backgrounds. RealWaste is more realistic but still modest in size and drawn from a landfill context. Neither dataset guarantees reliable classification of every Google image or every object in a cluttered scene.

A useful next experiment is to compare both backbones on RealWaste, analyze class-wise errors, and separately evaluate an external challenge set. If the intended task changes to finding several pieces of litter in a scene, use an object-detection dataset such as [TACO](https://github.com/pedropro/TACO) rather than treating the entire scene as one classification sample.

## Data acknowledgement

Cite TrashNet and RealWaste in any report, and review the source datasets' reuse terms before redistributing any images. This repository does not bundle datasets or pretrained checkpoints.
