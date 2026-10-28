# Waste Classification Lab

<p align="center">
  <strong>Waste Image Classification · MobileNetV2 · EfficientNetB0 · Gradio</strong>
</p>

A Python and TensorFlow project for classifying waste images with transfer-learning models. The application provides single-image and batch prediction, model comparison, evaluation reports, and a browser-based interface.

[GitHub repository](https://github.com/ak-himself/clgproject) · [TrashNet dataset](https://github.com/garythung/trashnet)

---

## Contents

- [Team and responsibilities](#team-and-responsibilities)
- [Project workflow](#project-workflow)
- [Repository structure](#repository-structure)
- [File reference](#file-reference)
- [Setup](#setup)
- [Prepare data](#prepare-data)
- [Train models](#train-models)
- [Evaluate models](#evaluate-models)
- [Run and test the GUI](#run-and-test-the-gui)
- [Temporary public access with Cloudflare Tunnel](#temporary-public-access-with-cloudflare-tunnel)
- [Git workflow](#git-workflow)
- [Limitations](#limitations)

## Team and responsibilities

Each member owns a primary work area. Integration and final verification are shared responsibilities.

### 1. Parteek — Project Lead & System Integration

**Enrollment:** `S24CSEU2232`

**Responsibilities**
- Coordinate milestones, task allocation, and progress.
- Maintain the overall project architecture and keep modules working together.
- Integrate preparation, training, evaluation, and GUI workflows.
- Review Git changes, resolve integration issues, and coordinate final testing.
- Maintain the project documentation and organize the final demonstration.

**Files used**
- `README.md` — main project guide.
- `.gitignore` — repository hygiene and exclusion rules.
- All modules in `src/` during integration and review.

### 2. Deep Jaiswal — Dataset & Data Preprocessing Engineer

**Enrollment:** `S24CSEU2561`

**Responsibilities**
- Maintain the expected image-folder and class-label structure.
- Check image readability, labels, duplicate files, and unexpected files.
- Generate data manifests and verify class counts before training.
- Confirm that training, validation, and test partitions are prepared correctly.
- Document dataset sources and avoid redistributing images without permission.

**Files used**
- `src/prepare_data.py` — scans image folders and generates manifests.
- `data/raw/dataset-resized/` — source image folders.
- `data/processed/` — generated manifests and class metadata.

### 3. Manasvi — MobileNetV2 Model Engineer

**Enrollment:** `S24CSEU0221`

**Responsibilities**
- Train and maintain the MobileNetV2 classifier.
- Tune training epochs and supported training options.
- Review accuracy, macro F1, per-class results, and inference time.
- Inspect the confusion matrix and investigate misclassified classes.
- Keep the model checkpoint and metadata consistent.

**Files used**
- `src/train.py` — model construction and training.
- `src/evaluate.py` — evaluation and report generation.
- `models/mobilenetv2_best.keras` — saved model checkpoint.
- `models/mobilenetv2_metadata.json` — model metadata.
- MobileNetV2 reports under `reports/`.

### 4. Ankit Kumar — EfficientNetB0 Model Engineer

**Enrollment:** `S24CSEU0910`

**Responsibilities**
- Train and maintain the EfficientNetB0 classifier.
- Tune training epochs and batch size where appropriate.
- Review accuracy, macro F1, per-class results, and inference time.
- Compare EfficientNetB0 results with MobileNetV2.
- Keep the model checkpoint and metadata consistent.

**Files used**
- `src/train.py` — shared model construction and training pipeline.
- `src/evaluate.py` — evaluation and report generation.
- `models/efficientnetb0_best.keras` — saved model checkpoint.
- `models/efficientnetb0_metadata.json` — model metadata.
- EfficientNetB0 reports under `reports/`.

### 5. Vansh Jain — Evaluation, GUI & Deployment Engineer

**Enrollment:** `S24CSEU0161`

**Responsibilities**
- Run evaluation and compare both model architectures.
- Test single-image upload, batch upload, model selection, and model comparison.
- Verify displayed predictions, scores, inference latency, and error messages.
- Run the Gradio application and expose it temporarily through Cloudflare Tunnel for demonstrations.
- Explain evaluation results and the security implications of public access.

**Files used**
- `src/app.py` — Gradio interface and model inference.
- `src/evaluate.py` — metrics and confusion matrices.
- `src/image_registry.py` — image-pixel hashing and exact-image lookup utility.
- `reports/` — evaluation outputs.
- `requirements.txt` — application dependencies.

---

## Project workflow

1. **Organize images:** place each image in the folder for its ground-truth class.
2. **Prepare data:** generate CSV manifests and class metadata.
3. **Train:** train MobileNetV2 and EfficientNetB0 independently.
4. **Evaluate:** calculate classification metrics and generate confusion matrices.
5. **Launch the interface:** upload one or more images and inspect predictions.
6. **Compare models:** run both architectures on the same input and compare results.
7. **Demonstrate:** if remote access is needed, expose the local Gradio port through a temporary Cloudflare Tunnel.

## Repository structure

The following tree shows the main source files and generated artifacts. Data, model, and report files are created locally by the workflow and may be excluded from Git.

```text
clgproject/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── app.py
│   ├── image_registry.py
│   ├── prepare_data.py
│   ├── train.py
│   └── evaluate.py
├── data/
│   ├── raw/
│   │   └── dataset-resized/
│   │       ├── cardboard/
│   │       ├── glass/
│   │       ├── metal/
│   │       ├── paper/
│   │       ├── plastic/
│   │       └── trash/
│   └── processed/
│       ├── train.csv
│       ├── val.csv
│       ├── test.csv
│       ├── class_names.json
│       └── dataset_metadata.json
├── models/
│   ├── mobilenetv2_best.keras
│   ├── mobilenetv2_metadata.json
│   ├── efficientnetb0_best.keras
│   └── efficientnetb0_metadata.json
└── reports/
    ├── mobilenetv2_metrics.json
    ├── mobilenetv2_confusion_matrix.png
    ├── efficientnetb0_metrics.json
    └── efficientnetb0_confusion_matrix.png
```

The tree is a guide to the main workflow, not a guarantee that every generated file is committed to the repository.

---

## File reference

### Source code

| File | Purpose | Primary owner(s) |
|---|---|---|
| `src/app.py` | Creates the Gradio web interface. Loads model checkpoints and metadata, prepares uploaded images, runs predictions, displays scores and latency, supports batch predictions, and compares model outputs. | Vansh, Parteek |
| `src/image_registry.py` | Applies EXIF orientation correction, converts images to RGB, and computes a SHA-256 hash from image dimensions and pixel data for exact-image lookup. | Deep, Vansh |
| `src/prepare_data.py` | Scans class folders, normalizes labels, generates CSV manifests, saves class ordering and dataset metadata, and prepares training/validation/test partitions. | Deep, Parteek |
| `src/train.py` | Loads the training manifest, decodes and resizes images to 224×224, builds MobileNetV2 or EfficientNetB0 with a classification head, trains the model, optionally fine-tunes backbone layers, and saves checkpoints plus metadata. | Manasvi, Ankit |
| `src/evaluate.py` | Loads a trained model, runs batch predictions, calculates accuracy and macro F1, produces per-class metrics, measures inference latency, and saves JSON reports and confusion-matrix plots. | Manasvi, Ankit, Vansh |

### Configuration and data

| File or directory | Purpose |
|---|---|
| `README.md` | Project overview, team responsibilities, file guide, commands, and limitations. |
| `requirements.txt` | Python dependencies for TensorFlow, NumPy, pandas, scikit-learn, Matplotlib, Pillow, and Gradio. |
| `.gitignore` | Excludes configured local environments, caches, datasets, and temporary files from Git. Review its rules before committing data or generated artifacts. |
| `data/raw/dataset-resized/` | Source image folders, one directory per class. |
| `data/processed/train.csv` | Paths and labels for training images. |
| `data/processed/val.csv` | Paths and labels for validation images. |
| `data/processed/test.csv` | Paths and labels for held-out test images. |
| `data/processed/class_names.json` | Ordered class names used to map model output indices to labels. |
| `data/processed/dataset_metadata.json` | Dataset source, class counts, and split configuration. |
| `models/*_best.keras` | Saved Keras model checkpoints. |
| `models/*_metadata.json` | Model identity, class order, image size, and related inference metadata. |
| `reports/*_metrics.json` | Accuracy, macro F1, sample count, inference time, and per-class metrics. |
| `reports/*_confusion_matrix.png` | Plot comparing true labels with predicted labels. |

---

## Setup

Commands assume CachyOS, Fish shell, pyenv Python 3.12.8, and the repository path `/home/ak/Code/clgproject`. Adjust these if your environment differs.

```fish
cd /home/ak/Code/clgproject
pyenv shell 3.12.8
python --version
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

A dedicated virtual environment is recommended for reproducible dependencies.

Check Python syntax:

```fish
python -m py_compile src/image_registry.py src/prepare_data.py src/train.py src/evaluate.py src/app.py
```

## Prepare data

Run commands from the repository root.

```fish
python src/prepare_data.py --dataset trashnet
```

The script reads images from `data/raw/dataset-resized/<class>/` and creates stratified training, validation, and test manifests under `data/processed/`. Inspect the printed class counts and check the labels before training.

## Train models

### MobileNetV2

```fish
python src/train.py --dataset trashnet --model mobilenetv2 --head-epochs 5 --fine-tune-epochs 3
```

### EfficientNetB0

```fish
python src/train.py --dataset trashnet --model efficientnetb0 --head-epochs 5 --fine-tune-epochs 3
```

### Training options

- `--dataset`: dataset identifier accepted by the script.
- `--model`: `mobilenetv2` or `efficientnetb0`.
- `--head-epochs`: epochs for training the classification head.
- `--fine-tune-epochs`: epochs for fine-tuning the final backbone layers; use `0` to skip.
- `--batch-size`: images per batch; default is `16`.

Training saves the best model checkpoint and a JSON metadata file under `models/`.

## Evaluate models

### MobileNetV2

```fish
python src/evaluate.py --dataset trashnet --model mobilenetv2
```

### EfficientNetB0

```fish
python src/evaluate.py --dataset trashnet --model efficientnetb0
```

Reports and confusion matrices are saved under `reports/`. Use the held-out test metrics to compare model performance; do not treat training accuracy as evidence of performance on unseen images.

## Run and test the GUI

Start the app:

```fish
cd /home/ak/Code/clgproject
pyenv shell 3.12.8
python src/app.py
```

Open the local URL printed by Gradio, normally `http://127.0.0.1:7860`.

### GUI test checklist

1. Select a model architecture.
2. Upload a single image and run classification.
3. Upload multiple images and check the batch results.
4. Enable model comparison and inspect the results from both architectures.
5. Review predicted labels, scores, top predictions, latency, and any loading errors.
6. After retraining, stop the application with `Ctrl+C` and restart it so the new checkpoints are loaded.

---

## Temporary public access with Cloudflare Tunnel

A Cloudflare Quick Tunnel creates a temporary public URL without requiring a configured domain or named tunnel.

### Terminal 1 — start the application

```fish
cd /home/ak/Code/clgproject
pyenv shell 3.12.8
python src/app.py
```

### Terminal 2 — start the tunnel

Keep the Gradio app running, open another terminal, and run:

```fish
cloudflared tunnel --url http://127.0.0.1:7860
```

Wait for the terminal to display a public URL ending in `.trycloudflare.com`, then open it on the device used for the demonstration. Keep both terminals running while the service is needed.

### Checks and shutdown

Check that `cloudflared` is installed:

```fish
cloudflared --version
```

Check that Gradio responds locally:

```fish
curl -I http://127.0.0.1:7860
```

Stop the tunnel with `Ctrl+C` in Terminal 2. Stop Gradio with `Ctrl+C` in Terminal 1. A Quick Tunnel usually receives a different random URL after restarting. A stable domain requires a domain managed through Cloudflare and a configured named tunnel.

**Security:** Anyone who obtains the public URL may be able to access the application. Avoid exposing confidential files or images, and stop the tunnel when the demonstration ends.

## Git workflow

Check your changes:

```fish
git status --short
git diff --check
```

Pull the latest shared changes:

```fish
git pull --ff-only origin main
```

Commit and push the README update after reviewing it:

```fish
git add README.md
git commit -m "Update public project documentation"
git push origin main
```

Stage only the files you intend to commit. Do not commit credentials, private data, or images you do not have permission to redistribute.

## Limitations

- The classifier predicts one label for the whole image; it does not locate multiple objects in a scene.
- A model score is not a guarantee of correctness.
- Performance depends on the quality, size, and diversity of the labelled data.
- Exact-image lookup only matches the same canonical pixel content and dimensions; resizing, cropping, recompressing, or editing can prevent a match.
- CPU training and inference may be slower than GPU-backed execution.
- Cloudflare Quick Tunnel URLs are temporary and are intended for demonstrations, not permanent production hosting.

## Dataset acknowledgement

This project uses the [TrashNet dataset](https://github.com/garythung/trashnet). Cite the dataset in reports and review its reuse terms. Do not redistribute third-party images unless their license or permission allows it.
