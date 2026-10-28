# Waste Classification with Transfer Learning

Train and compare MobileNetV2 and EfficientNetB0 on TrashNet, evaluate them on a held-out test split, and test new images through a Gradio GUI.

## Stack

- Python 3.12
- TensorFlow/Keras (CPU is supported)
- MobileNetV2 and EfficientNetB0 pretrained on ImageNet
- scikit-learn, pandas, Matplotlib
- Gradio image-upload interface

## Dataset

This project uses the six-class [TrashNet dataset](https://github.com/garythung/trashnet): cardboard, glass, metal, paper, plastic, and trash.

Download the resized dataset archive and extract it so the folder layout is:

```text
data/raw/dataset-resized/
├── cardboard/
├── glass/
├── metal/
├── paper/
├── plastic/
└── trash/
```

The dataset and all generated model files are intentionally excluded from Git.

## Setup (CachyOS / Linux, fish shell)

Activate your existing Python 3.12.8 installation:

```fish
cd /home/ak/Code/clgproject
pyenv shell 3.12.8
```

For reproducibility, a dedicated virtual environment is recommended. If you choose to use one, create it with Python 3.12.8 and activate it before installing dependencies. Do not install project dependencies into system Python.

```fish
python -m venv .venv
source .venv/bin/activate.fish
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 1. Prepare the dataset

After extracting the archive:

```fish
python src/prepare_data.py
```

This creates deterministic, stratified CSV manifests under `data/processed/` with approximately 70% training, 15% validation, and 15% test images. Images are not duplicated. The test set is held out from model training and selection.

## 2. Train models

Train MobileNetV2 first:

```fish
python src/train.py --model mobilenetv2 --head-epochs 5 --fine-tune-epochs 5
```

Then train EfficientNetB0 using the same data split:

```fish
python src/train.py --model efficientnetb0 --head-epochs 5 --fine-tune-epochs 5
```

The first run downloads ImageNet pretrained weights, so an internet connection is required. CPU training can take some time. Reduce epochs or use `--fine-tune-epochs 0` for a faster initial smoke test.

Best checkpoints are saved under `models/`, which is ignored by Git.

## 3. Evaluate on the held-out test set

```fish
python src/evaluate.py --model mobilenetv2
python src/evaluate.py --model efficientnetb0
```

Metrics and confusion matrices are written to `reports/`. Compare accuracy and macro F1 rather than accuracy alone because TrashNet has class imbalance.

## 4. Upload an image in the GUI

After training at least one model:

```fish
python src/app.py
```

Open the local URL printed by Gradio, choose a trained model, upload a JPG/PNG image, and click **Classify image**.

## Project structure

```text
src/
├── prepare_data.py
├── train.py
├── evaluate.py
└── app.py
data/       # ignored: raw images and generated split manifests
models/     # ignored: trained checkpoints
reports/    # ignored: generated evaluation outputs
```

## Evaluation caveats

TrashNet is a small dataset with relatively plain backgrounds and an imbalanced class distribution. A strong score on its test split does not guarantee reliable classification of arbitrary real-world waste photographs. Keep the test set untouched until final evaluation, and test additional real-world images separately.

## License and data

Review the dataset's license and terms on its original source before redistributing images. This project does not bundle the dataset.
