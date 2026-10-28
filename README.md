# Waste Classification Lab

<p align="center">
  <strong>TrashNet + Custom Dataset · MobileNetV2 + EfficientNetB0 · Gradio</strong>
</p>

A Python and TensorFlow project for classifying waste images. The application lets users select a dataset and model, classify one or more images, and compare both model architectures.

[Repository](https://github.com/ak-himself/clgproject) · [TrashNet source](https://github.com/garythung/trashnet)

---

## Contents

- [Team and ownership](#team-and-ownership)
- [Project workflow](#project-workflow)
- [Repository structure](#repository-structure)
- [File reference](#file-reference)
- [Setup](#setup)
- [Prepare datasets](#prepare-datasets)
- [Train models](#train-models)
- [Evaluate models](#evaluate-models)
- [Run the GUI](#run-the-gui)
- [Share the GUI with Cloudflare Tunnel](#share-the-gui-with-cloudflare-tunnel)
- [Git workflow](#git-workflow)
- [Limitations and interpretation](#limitations-and-interpretation)

## Team and ownership

The project has five members. Each role has a primary area of responsibility; integration and final verification are shared.

### 1. Parteek — Project Lead & System Integration

**Enrollment:** `S24CSEU2232`

**Responsibilities**
- Coordinate milestones, divide tasks, and track completion.
- Maintain the overall architecture and ensure TrashNet and Custom remain separate.
- Integrate dataset preparation, training, evaluation, GUI, and deployment workflows.
- Review repository changes, resolve integration issues, and coordinate final testing.
- Maintain project documentation and prepare the final demonstration.

**Primary files**
- `README.md` — project documentation and workflow.
- `CUSTOM_DATASET_WORKFLOW.md` — quick reference for the Custom workflow.
- `.gitignore` — repository hygiene and local-data exclusions.
- All files under `src/` during integration and review.

### 2. Deep Jaiswal — Dataset & Data Preprocessing Engineer

**Enrollment:** `S24CSEU2561`

**Responsibilities**
- Maintain the expected class-folder structures for TrashNet and Custom.
- Check image readability, class labels, duplicates, and unexpected files.
- Prepare manifests and verify per-class image counts before training.
- Keep the TrashNet train/validation/test split separate from the all-images Custom workflow.
- Document dataset sources and avoid redistributing images without permission.

**Primary files**
- `src/prepare_data.py` — scans image folders and generates manifests.
- `src/image_registry.py` — builds the exact-pixel lookup for Custom images.
- `data/raw/dataset-resized/` — local TrashNet source images.
- `data/custom_images/train/` — Custom class folders.
- `data/processed/` — generated manifests and class metadata.

### 3. Manasvi — MobileNetV2 Model Engineer

**Enrollment:** `S24CSEU0221`

**Responsibilities**
- Train and maintain the MobileNetV2 model for both datasets.
- Tune head-training and fine-tuning epochs using the supported command-line options.
- Inspect Custom training-set memorization results and TrashNet held-out test results.
- Review per-class metrics and confusion matrices, and record misclassifications.
- Keep checkpoint files and metadata consistent with the selected dataset.

**Primary files**
- `src/train.py` — shared model-building and training pipeline.
- `src/evaluate.py` — model evaluation and report generation.
- `models/mobilenetv2_best.keras` and `models/mobilenetv2_metadata.json` — TrashNet artifacts.
- `models/custom_mobilenetv2_best.keras` and `models/custom_mobilenetv2_metadata.json` — Custom artifacts.
- Matching MobileNetV2 reports under `reports/`.

### 4. Ankit Kumar — EfficientNetB0 Model Engineer

**Enrollment:** `S24CSEU0910`

**Responsibilities**
- Train and maintain the EfficientNetB0 model for both datasets.
- Tune training epochs and batch size where appropriate.
- Inspect Custom training-set memorization results and TrashNet held-out test results.
- Compare EfficientNetB0 metrics and inference time with MobileNetV2.
- Ensure dataset-specific checkpoints and metadata are saved to the correct paths.

**Primary files**
- `src/train.py` — shared training pipeline, including EfficientNetB0.
- `src/evaluate.py` — metrics and confusion-matrix generation.
- `models/efficientnetb0_best.keras` and `models/efficientnetb0_metadata.json` — TrashNet artifacts.
- `models/custom_efficientnetb0_best.keras` and `models/custom_efficientnetb0_metadata.json` — Custom artifacts.
- Matching EfficientNetB0 reports under `reports/`.

### 5. Vansh Jain — Evaluation, GUI & Deployment Engineer

**Enrollment:** `S24CSEU0161`

**Responsibilities**
- Run and compare evaluation results for both architectures and datasets.
- Test dataset selection, model selection, single-image uploads, batch uploads, and model comparison.
- Verify that Custom exact-image registry results are shown separately from neural-network predictions.
- Check inference latency, report output, and missing-model error handling.
- Run the Gradio interface and expose it temporarily through Cloudflare Tunnel for demonstrations.
- Explain the limitations and security implications of a public temporary URL.

**Primary files**
- `src/evaluate.py` — evaluation metrics and confusion matrices.
- `src/app.py` — Gradio interface and model inference.
- `src/image_registry.py` — exact-pixel matching used in Custom mode.
- `reports/` — evaluation results.
- `requirements.txt` — application dependencies.

---

## Project workflow

1. **Organize and label images.** Place each image in its intended class folder.
2. **Prepare the manifests.** Run `src/prepare_data.py` to generate CSVs and class metadata.
3. **Train each architecture separately.** Train MobileNetV2 and EfficientNetB0 for TrashNet and Custom. Their checkpoints use different names.
4. **Evaluate.** Custom is measured against the same images used for training; TrashNet is measured against its held-out test set.
5. **Launch the GUI.** Select the dataset and model, upload images, and inspect predictions or compare both models.
6. **Test Custom exact matching.** The registry looks for the same canonical decoded RGB pixels as a registered image.
7. **Demonstrate remotely if needed.** Start a temporary Cloudflare Tunnel while the local app is running.

### Dataset behaviour

| Setting | TrashNet | Custom |
|---|---|---|
| Source directory | `data/raw/dataset-resized/` | `data/custom_images/train/` |
| Training data | Approximately 70% split | 100% of labelled images |
| Validation data | Approximately 15% split | No separate split |
| Test data | Approximately 15% held out | No held-out test set |
| Evaluation meaning | Held-out test performance | Training-set memorization check |
| Checkpoint prefix | No prefix | `custom_` |

**Custom is deliberately a memorization workflow.** All labelled Custom images are used for training; Custom evaluation therefore measures fit to those same images and does not demonstrate generalization to unseen images. More epochs do not guarantee 100% neural-network accuracy.

The current Custom collection has 10 images in each of five folders—cardboard, glass, metal, paper, and plastic—for 50 images total. Its `trash/` folder is empty. The Custom neural networks therefore learn five classes, not a learned trash class.

In the GUI, the Custom exact-image registry is a separate mechanism: it returns the folder label for a registered image whose orientation-corrected RGB pixels and dimensions match exactly. Filename and file metadata do not determine the match. Resizing, cropping, recompressing, or editing pixels can break the match. An unregistered image is assigned `trash` by the requested fallback policy; this does not prove that the image is actually trash.

---

## Repository structure

The tree below shows the main source files and the locations where datasets and generated artifacts are expected. Some directories and files are created only after running the preparation, training, or evaluation commands.

```text
clgproject/
├── README.md
├── CUSTOM_DATASET_WORKFLOW.md
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
│   ├── custom_images/
│   │   └── train/
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
│       ├── dataset_metadata.json
│       └── custom/
│           ├── train.csv
│           ├── class_names.json
│           └── dataset_metadata.json
├── models/
│   ├── mobilenetv2_best.keras
│   ├── mobilenetv2_metadata.json
│   ├── efficientnetb0_best.keras
│   ├── efficientnetb0_metadata.json
│   ├── custom_mobilenetv2_best.keras
│   ├── custom_mobilenetv2_metadata.json
│   ├── custom_efficientnetb0_best.keras
│   └── custom_efficientnetb0_metadata.json
└── reports/
    ├── mobilenetv2_metrics.json
    ├── mobilenetv2_confusion_matrix.png
    ├── efficientnetb0_metrics.json
    ├── efficientnetb0_confusion_matrix.png
    ├── custom_mobilenetv2_metrics.json
    ├── custom_mobilenetv2_confusion_matrix.png
    ├── custom_efficientnetb0_metrics.json
    └── custom_efficientnetb0_confusion_matrix.png
```

---

## File reference

### Source code

| File | What it does | Main users |
|---|---|---|
| `src/app.py` | Creates the Gradio interface. Loads available dataset/model checkpoints and metadata, preprocesses uploaded images, runs predictions, displays scores and latency, supports batch predictions and comparing both models, and shows the Custom registry result separately. | Vansh, Parteek |
| `src/image_registry.py` | Canonicalizes an image with EXIF orientation correction and RGB conversion, hashes its dimensions and pixel bytes using SHA-256, and builds a hash-to-label map from Custom class folders. If identical pixels are assigned conflicting labels, the registry maps that hash to `trash`. | Deep, Vansh |
| `src/prepare_data.py` | Scans the dataset folders, normalizes folder names into labels, writes CSV manifests, saves class order and dataset metadata, and applies the dataset-specific preparation policy. TrashNet is stratified into train/validation/test; Custom puts all labelled images into `train.csv` and deletes stale Custom `val.csv`/`test.csv` files. | Deep, Parteek |
| `src/train.py` | Loads the training manifest, decodes and resizes images to 224×224, builds MobileNetV2 or EfficientNetB0 with a classification head, trains the head and optionally fine-tunes the final backbone layers, and saves the best checkpoint and metadata. Custom mode disables augmentation and dropout. | Manasvi, Ankit |
| `src/evaluate.py` | Loads a checkpoint and manifest, predicts in batches, calculates accuracy, macro F1 and per-class metrics, measures mean inference latency, and writes a JSON report and confusion-matrix image. Custom evaluates on its training manifest; TrashNet evaluates on the held-out test manifest. | Manasvi, Ankit, Vansh |

### Configuration and documentation

| File or directory | What it contains |
|---|---|
| `README.md` | Main project guide: team ownership, workflow, file descriptions, setup, commands, and limitations. |
| `CUSTOM_DATASET_WORKFLOW.md` | Short operational guide focused on the Custom memorization workflow. |
| `requirements.txt` | Python dependencies: TensorFlow, NumPy, pandas, scikit-learn, Matplotlib, Pillow, and Gradio. |
| `.gitignore` | Rules for excluding local environments, caches, selected dataset paths, and temporary artifacts from Git. Always check it before adding large image collections. |
| `data/raw/dataset-resized/` | Local TrashNet image folders. |
| `data/custom_images/train/` | Local Custom image folders. |
| `data/processed/` | Generated CSV manifests, class ordering, and dataset metadata. |
| `models/` | Keras checkpoints and per-model metadata. TrashNet files have no dataset prefix; Custom files start with `custom_`. |
| `reports/` | JSON evaluation metrics and PNG confusion matrices for each dataset/model combination. |

### Generated files

| Artifact | Purpose |
|---|---|
| `train.csv`, `val.csv`, `test.csv` | TrashNet paths and ground-truth labels for its three splits. |
| `custom/train.csv` | All labelled Custom image paths and labels, with no held-out split. |
| `class_names.json` | Ordered class names used to map model output indices back to labels. |
| `dataset_metadata.json` | Dataset name, source path, class counts, and split configuration. |
| `*_best.keras` | Saved trained Keras model. |
| `*_metadata.json` | Dataset/model identity, class order, image size, and other model metadata used by the GUI. |
| `*_metrics.json` | Evaluation type, sample count, accuracy, macro F1, inference time, and per-class report. |
| `*_confusion_matrix.png` | Plot of true labels versus predicted labels. |

---

## Setup

Commands below assume CachyOS, Fish shell, pyenv Python 3.12.8, and the project directory `/home/ak/Code/clgproject`. Change the path or Python version if your setup differs.

```fish
cd /home/ak/Code/clgproject
pyenv shell 3.12.8
python --version
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

A dedicated virtual environment is recommended for reproducible project dependencies. The commands above use the established local pyenv interpreter.

Check syntax for all Python modules:

```fish
python -m py_compile src/image_registry.py src/prepare_data.py src/train.py src/evaluate.py src/app.py
```

---

## Prepare datasets

Run commands from the repository root.

### Custom

```fish
python src/prepare_data.py --dataset custom
```

Reads images from `data/custom_images/train/<class>/` and writes the complete training manifest to `data/processed/custom/train.csv`. Every labelled image is included.

### TrashNet

```fish
python src/prepare_data.py --dataset trashnet
```

Reads images from `data/raw/dataset-resized/<class>/` and writes the stratified train, validation, and test manifests under `data/processed/`.

Inspect class counts and confirm the labels before training. Folder names are used as ground-truth labels.

---

## Train models

### Custom: train on all labelled images

```fish
python src/train.py --dataset custom --model mobilenetv2 --head-epochs 40 --fine-tune-epochs 30
python src/train.py --dataset custom --model efficientnetb0 --head-epochs 40 --fine-tune-epochs 30
```

### TrashNet: standard training

These commands train or replace the existing TrashNet checkpoints:

```fish
python src/train.py --dataset trashnet --model mobilenetv2 --head-epochs 5 --fine-tune-epochs 3
python src/train.py --dataset trashnet --model efficientnetb0 --head-epochs 5 --fine-tune-epochs 3
```

### Training options

- `--dataset`: `custom` or `trashnet`.
- `--model`: `mobilenetv2` or `efficientnetb0`.
- `--head-epochs`: number of epochs for training the classification head.
- `--fine-tune-epochs`: number of fine-tuning epochs; set to `0` to skip fine-tuning.
- `--batch-size`: images per batch; default is `16`.

TrashNet training uses augmentation and validation-based early stopping. Custom mode uses all Custom images and disables augmentation/dropout to make memorization easier. The reported Custom monitoring accuracy is measured on those same training images.

---

## Evaluate models

### Custom: training-set memorization check

```fish
python src/evaluate.py --dataset custom --model mobilenetv2
python src/evaluate.py --dataset custom --model efficientnetb0
```

### TrashNet: held-out test evaluation

```fish
python src/evaluate.py --dataset trashnet --model mobilenetv2
python src/evaluate.py --dataset trashnet --model efficientnetb0
```

Reports and confusion matrices are written under `reports/`. Custom report names start with `custom_`. The exact registry self-check is reported separately and is **not** neural-network accuracy.

---

## Run the GUI

Start the app in **Terminal 1**:

```fish
cd /home/ak/Code/clgproject
pyenv shell 3.12.8
python src/app.py
```

Open the local URL printed by Gradio, normally `http://127.0.0.1:7860`.

### GUI test checklist

1. Select **TrashNet** or **Custom**.
2. Select **MobileNetV2** or **EfficientNetB0**.
3. Enable **Compare both models** to compare both backbones for the selected dataset.
4. Upload one image and run classification.
5. Upload multiple images and check the batch results table.
6. In Custom mode, compare the strict registry label with each neural-network prediction.
7. Verify scores, top predictions, latency, and any model-loading error message.

The GUI loads models when it starts. After retraining, stop the old process with `Ctrl+C` and run `python src/app.py` again to load the updated checkpoints.

---

## Share the GUI with Cloudflare Tunnel

A Quick Tunnel provides a temporary public URL without requiring a configured domain or a named tunnel.

### Terminal 1 — run Gradio

```fish
cd /home/ak/Code/clgproject
pyenv shell 3.12.8
python src/app.py
```

### Terminal 2 — create the tunnel

Keep Gradio running, open a second terminal, and run:

```fish
cloudflared tunnel --url http://127.0.0.1:7860
```

Wait for `cloudflared` to print a public URL ending in `.trycloudflare.com`, then open that URL on the demonstration device. Keep both terminals open for as long as the app is needed.

### Troubleshooting and stopping

Check that Cloudflare Tunnel is installed:

```fish
cloudflared --version
```

Check that the local Gradio server responds:

```fish
curl -I http://127.0.0.1:7860
```

Stop the tunnel with `Ctrl+C` in Terminal 2. Stop Gradio with `Ctrl+C` in Terminal 1. Restarting a Quick Tunnel generally creates a different random URL. A stable custom hostname requires a Cloudflare-managed domain and a configured named tunnel.

**Security:** Anyone who obtains the public URL may be able to access the application. Use the temporary tunnel only for suitable demonstrations, do not upload confidential images, and stop the tunnel when finished.

---

## Git workflow

Check your changes before committing:

```fish
git status --short
git diff --check
```

Pull shared changes:

```fish
git pull --ff-only origin main
```

After reviewing the changes, commit and push the documentation:

```fish
git add README.md
git commit -m "Improve project README and file documentation"
git push origin main
```

If other files changed, stage only the intended files. Do not commit API keys, private files, or image datasets that you are not permitted to redistribute.

---

## Limitations and interpretation

- The application classifies the whole image; it does not detect and label multiple objects within one scene.
- Model scores are not guarantees of correctness.
- Custom training accuracy measures fit to the training images and does not establish generalization to unseen, resized, edited, or recompressed photos.
- Custom currently has no labelled `trash` training examples. Its strict registry fallback is a fixed policy, not a learned trash prediction.
- Exact-image registry matching depends on decoded RGB pixel content and dimensions.
- CPU training and inference may be slower than GPU-backed execution.
- Cloudflare Quick Tunnel URLs are temporary and should not be treated as permanent production hosting.

## Dataset acknowledgement

The project uses [TrashNet](https://github.com/garythung/trashnet). Cite the dataset in reports and review its reuse terms. Do not redistribute third-party images unless their license or permission allows it.
