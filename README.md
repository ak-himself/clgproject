# Waste Classification Lab

A Python/TensorFlow waste-image classification project with two independent datasets and two transfer-learning backbones:

- **Datasets:** TrashNet and Custom
- **Models:** MobileNetV2 and EfficientNetB0
- **Interface:** Gradio GUI for single-image, batch-image, and model-comparison workflows

The Custom dataset is intended to fit a fixed, manually labelled set of images. It uses every labelled image for training rather than reserving a validation or test split. This is a deliberate memorization experiment, not a claim of generalization to unseen images.

## Features

- Select TrashNet or Custom independently of the model choice.
- Select MobileNetV2 or EfficientNetB0.
- Compare both models on the same input.
- Upload one image or a batch and inspect model predictions, scores, and latency.
- In Custom mode, use an exact-image registry to return the saved folder label for a registered image; unregistered images fall back to \`trash\` according to the requested policy.
- Keep TrashNet and Custom checkpoints, metadata, manifests, and reports separate.

## Datasets

### TrashNet

The [original TrashNet dataset](https://github.com/garythung/trashnet) contains 2,527 images across six classes: cardboard, glass, metal, paper, plastic, and trash.

Expected local path:

\`\`\`text
data/raw/dataset-resized/
├── cardboard/
├── glass/
├── metal/
├── paper/
├── plastic/
└── trash/
\`\`\`

TrashNet uses stratified splits of approximately 70% training, 15% validation, and 15% testing. Its existing manifests are stored in \`data/processed/\`.

### Custom

The Custom images live under:

\`\`\`text
data/custom_images/train/
├── cardboard/
├── glass/
├── metal/
├── paper/
├── plastic/
└── trash/
\`\`\`

Place each image in the folder matching its intended label. The current curated set has 10 images in each of the five non-trash folders (50 images total); the \`trash/\` folder is empty. Therefore, the Custom neural networks currently learn five classes, not a learned trash class.

Custom mode uses **100% of its labelled images**. It does not create a held-out validation or test split. The reported Custom accuracy is calculated on those same training images and must be described as **training-set memorization accuracy**, not test accuracy or generalization performance.

The GUI also has a strict exact-image rule: it hashes orientation-corrected, decoded RGB pixels and maps an exact match to its folder label. Filenames and file metadata do not matter, but resizing, cropping, recompressing, or editing pixels can prevent an exact match. Any image that does not match a registered image is assigned \`trash\` by policy; that does not prove the image is actually trash. The strict registry result is shown separately from the neural-network prediction.

### Data privacy and repository size

Keep downloaded datasets and external images local unless you have permission to redistribute them. Check \`.gitignore\` before adding image collections or model artifacts to Git.

## Environment

The project has been used with Python 3.12.8 via pyenv on CachyOS. TensorFlow can run on CPU when a supported NVIDIA CUDA GPU is unavailable.

\`\`\`fish
cd /home/ak/Code/clgproject
pyenv shell 3.12.8
python -m pip install -r requirements.txt
\`\`\`

A dedicated virtual environment is preferable for repeatability. Avoid installing project packages into system Python.

## Prepare the datasets

Run these commands from the repository root:

\`\`\`fish
# Prepare the standard TrashNet splits
python src/prepare_data.py --dataset trashnet

# Prepare Custom manifests using every labelled image
python src/prepare_data.py --dataset custom
\`\`\`

TrashNet manifests are written to \`data/processed/\`. Custom's full training manifest and class metadata are written to \`data/processed/custom/\`; old Custom validation/test manifests are removed so they cannot be mistaken for valid held-out data.

## Train models

### Custom — train on all labelled images

The following commands use every Custom image and disable augmentation and dropout to make memorization easier:

\`\`\`fish
python src/train.py --dataset custom --model mobilenetv2 --head-epochs 40 --fine-tune-epochs 30
python src/train.py --dataset custom --model efficientnetb0 --head-epochs 40 --fine-tune-epochs 30
\`\`\`

Custom checkpoints and metadata use the \`custom_\` prefix:

- \`models/custom_mobilenetv2_best.keras\`
- \`models/custom_mobilenetv2_metadata.json\`
- \`models/custom_efficientnetb0_best.keras\`
- \`models/custom_efficientnetb0_metadata.json\`

Training for more epochs does not guarantee 100% neural-network accuracy. Check the actual training-set metrics after each run.

### TrashNet — standard training

Existing TrashNet checkpoints are kept under their original names. To intentionally train or replace a TrashNet checkpoint:

\`\`\`fish
python src/train.py --dataset trashnet --model mobilenetv2 --head-epochs 5 --fine-tune-epochs 3
python src/train.py --dataset trashnet --model efficientnetb0 --head-epochs 5 --fine-tune-epochs 3
\`\`\`

TrashNet uses the train/validation split, mild augmentation, class weighting, and early stopping.

## Evaluate

\`\`\`fish
# Custom: evaluates on all training images (memorization check)
python src/evaluate.py --dataset custom --model mobilenetv2
python src/evaluate.py --dataset custom --model efficientnetb0

# TrashNet: evaluates on the held-out test split
python src/evaluate.py --dataset trashnet --model mobilenetv2
python src/evaluate.py --dataset trashnet --model efficientnetb0
\`\`\`

Reports and confusion matrices are saved under \`reports/\`. Custom report files are prefixed with \`custom_\`. Custom evaluation explicitly labels its result as a training-set memorization check. The exact registry self-check only verifies that registered files map to their folder labels; it is not a neural-network accuracy metric.

In the latest run captured during development, Custom training-set accuracy was 98% for MobileNetV2 and 96% for EfficientNetB0. These are measurements from that run, not guaranteed or held-out performance; later training runs may differ.

## Launch the GUI

\`\`\`fish
cd /home/ak/Code/clgproject
pyenv shell 3.12.8
python src/app.py
\`\`\`

Open the local URL Gradio prints. The GUI offers:

- **Dataset:** TrashNet or Custom
- **Model:** MobileNetV2 or EfficientNetB0
- **Compare both models:** run both backbones against the same input
- Single-image and batch-image upload with a results table

Models are loaded when the application starts. Restart the application after retraining so the GUI loads the newly saved checkpoints. If a checkpoint or metadata is missing or cannot be loaded, the GUI reports the problem.

## Important limitations

- The classifier predicts one class for the whole image; it does not locate multiple objects.
- A model score is not a guarantee of correctness.
- The Custom neural networks currently have only five learned classes because there are no labelled trash examples.
- The strict Custom registry only guarantees the stored label for an exact decoded-pixel match. Its fallback-to-trash rule is a fixed policy for unmatched images, not semantic recognition.
- Fitting all Custom images can produce high training accuracy without making the model reliable on resized, edited, or unseen photos.

## Data acknowledgement

Cite TrashNet in reports and review the source dataset's reuse terms before redistributing images. Do not publish third-party images unless their license or permission allows redistribution. This repository does not need to bundle the full TrashNet dataset to run, provided the dataset is available locally.
