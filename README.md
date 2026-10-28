# Waste Classification Lab

A Python/TensorFlow image-classification project that compares two image datasets and two transfer-learning models through a Gradio web interface.

- **Datasets:** TrashNet and Custom
- **Models:** MobileNetV2 and EfficientNetB0
- **Interface:** Gradio GUI for single-image prediction, batch prediction, and model comparison
- **Runtime:** Python 3.12, TensorFlow, CPU-compatible

Repository: https://github.com/ak-himself/clgproject

---

## 1. Team members and responsibilities

### 1. Parteek — Project Lead & System Integration
**Enrollment:** S24CSEU2232

Responsibilities:
- Coordinate project milestones and divide work among team members.
- Maintain the overall architecture and keep the TrashNet and Custom workflows consistent.
- Integrate dataset preparation, training, evaluation, and GUI modules.
- Review Git changes, resolve merge/integration problems, and maintain the repository.
- Check that the application can be run from the documented commands.
- Coordinate the final demonstration and project documentation.

### 2. Deep Jaiswal — Dataset & Data Preprocessing Engineer
**Enrollment:** S24CSEU2561

Responsibilities:
- Maintain the expected folder structure for TrashNet and Custom images.
- Verify that each image is placed in the folder corresponding to its intended class.
- Check for unreadable images, duplicates, incorrect labels, and inconsistent class-folder names.
- Run the dataset preparation script and inspect generated manifests and class counts.
- Preserve the separation between the original TrashNet dataset and the curated Custom dataset.
- Document dataset sources and avoid redistributing third-party images without permission.

### 3. Manasvi — MobileNetV2 Model Engineer
**Enrollment:** S24CSEU0221

Responsibilities:
- Train and maintain the MobileNetV2 model for both datasets.
- Tune training epochs and other supported training options.
- Review training-set memorization results for Custom and held-out test results for TrashNet.
- Inspect MobileNetV2 reports and confusion matrices to identify misclassified classes.
- Keep the model checkpoint and metadata consistent with the selected dataset.
- Record the exact commands and results used in the final demonstration.

### 4. Ankit Kumar — EfficientNetB0 Model Engineer
**Enrollment:** S24CSEU0910

Responsibilities:
- Train and maintain the EfficientNetB0 model for both datasets.
- Tune training epochs and other supported training options.
- Review Custom training-set memorization results and TrashNet held-out test results.
- Inspect the EfficientNetB0 classification report and confusion matrix.
- Compare results and inference time with MobileNetV2.
- Ensure that EfficientNetB0 checkpoint files and metadata are saved under the correct dataset-specific names.

### 5. Vansh Jain — Evaluation, GUI & Deployment Engineer
**Enrollment:** S24CSEU0161

Responsibilities:
- Run evaluation scripts for both models and both datasets.
- Compare accuracy, macro F1, inference latency, and confusion matrices.
- Test single-image upload, batch-image upload, dataset selection, model selection, and the compare-both-models option.
- Verify that the Custom exact-image registry result is displayed separately from the neural-network prediction.
- Launch the Gradio application and create a temporary public URL with Cloudflare Tunnel for demonstrations.
- Test the app from the public URL and explain deployment limitations, including that a temporary tunnel URL changes when restarted.

---

## 2. Project workflow

The main workflow is:

1. **Collect and label images.** Keep each image inside its intended class folder.
2. **Prepare manifests.** The preparation script records image paths and labels. TrashNet is split into train/validation/test sets; Custom uses all labelled images for training.
3. **Train the models.** Train MobileNetV2 and EfficientNetB0 separately for each dataset. The model names and dataset names determine separate checkpoint paths.
4. **Evaluate.** Custom evaluation measures how closely each neural network fits the same images it trained on. TrashNet evaluation uses the held-out test split.
5. **Launch the GUI.** Select the dataset and model, upload one or more images, and inspect predictions.
6. **Use strict Custom matching when required.** In Custom mode, the exact-image registry returns the saved folder label when the uploaded image has the same canonical decoded RGB pixels as a registered image. An image without an exact match falls back to \`trash\` according to the project's chosen policy.
7. **Deploy temporarily.** Expose the local Gradio port with Cloudflare Tunnel for a demonstration.

### Important: what “100%” means in Custom mode

The Custom workflow intentionally uses 100% of the labelled images for training; it does not reserve a validation or test split. Custom evaluation therefore reports **training-set memorization accuracy**, not generalization or unseen-image accuracy.

The exact-image registry is a separate lookup mechanism. It can return the folder label for an exact pixel match regardless of the neural network's prediction. A resized, cropped, recompressed, or edited image may no longer match. Unmatched images are assigned \`trash\` by policy; this does not establish that they are actually trash.

The current Custom dataset has 10 images in each of the five non-trash folders (50 images total) and no labelled images in \`trash/\`. Thus, the Custom neural networks learn five classes; \`trash\` is currently the strict registry's fallback label, not a learned Custom class.

---

## 3. Repository structure and what each file does

\`\`\`text
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
│   │   └── dataset-resized/       # Local TrashNet class folders
│   ├── custom_images/
│   │   └── train/                 # Custom class folders
│   └── processed/
│       ├── train.csv              # TrashNet training manifest
│       ├── val.csv                # TrashNet validation manifest
│       ├── test.csv               # TrashNet test manifest
│       ├── class_names.json       # TrashNet class order
│       ├── dataset_metadata.json # TrashNet dataset/split summary
│       └── custom/
│           ├── train.csv          # All labelled Custom images
│           ├── class_names.json   # Custom class order
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
\`\`\`

The tree above describes the principal files and generated artifacts. Some files are created only after running preparation, training, or evaluation. Dataset images and model files may be ignored by Git or present locally depending on repository configuration.

### Source files

| File | Purpose |
|---|---|
| \`src/app.py\` | Builds and launches the Gradio interface. Loads the four dataset/model combinations when available, provides dataset and model dropdowns, single-image and batch inputs, optional comparison, prediction scores, and latency. In Custom mode it displays the strict registry result separately from model predictions. Restart the app after retraining to load updated checkpoints. |
| \`src/image_registry.py\` | Creates a lookup table for Custom images using a SHA-256 hash of orientation-corrected, decoded RGB pixels and image dimensions. Filenames and metadata are not used as identity. Conflicting labels for identical pixels are treated as \`trash\`. |
| \`src/prepare_data.py\` | Scans class folders, creates labelled CSV manifests, writes class-name and dataset metadata JSON files, and prepares data for training. TrashNet gets stratified train/validation/test splits; Custom puts all labelled images into \`train.csv\` and removes stale Custom validation/test CSVs. |
| \`src/train.py\` | Loads manifests and image data, builds MobileNetV2 or EfficientNetB0 transfer-learning models, trains the classification head, optionally fine-tunes the last backbone layers, and saves the best checkpoint plus metadata. Custom mode disables augmentation and dropout to encourage memorization. |
| \`src/evaluate.py\` | Loads a saved checkpoint, predicts labels, calculates accuracy, macro F1, classification-report metrics, and mean inference time, and saves a JSON report and confusion-matrix PNG. Custom uses all training images for a memorization check; TrashNet uses its held-out test manifest. |
| \`requirements.txt\` | Lists the Python packages required for TensorFlow training, image processing, metrics, plotting, and the Gradio GUI. |
| \`.gitignore\` | Excludes Python caches, environment files, selected local dataset folders and temporary artifacts. Check its rules before adding large image datasets or generated files. |
| \`CUSTOM_DATASET_WORKFLOW.md\` | Quick reference for the separate dataset workflow, training commands, evaluation commands, and Custom memorization behaviour. |
| \`README.md\` | Main project documentation: team responsibilities, architecture, file guide, setup, commands, and limitations. |

### Generated data and model artifacts

- \`data/processed/train.csv\`, \`val.csv\`, and \`test.csv\`: TrashNet image paths and labels for the three splits.
- \`data/processed/custom/train.csv\`: all labelled Custom image paths and labels; no holdout split is used.
- \`class_names.json\`: class ordering used to translate model output indices into labels.
- \`dataset_metadata.json\`: dataset source, class counts, and split configuration.
- \`models/*_best.keras\`: trained Keras model checkpoints.
- \`models/*_metadata.json\`: dataset name, model name, class ordering, image size, checkpoint path, and training-mode information.
- \`reports/*_metrics.json\`: numeric evaluation metrics and per-class classification report.
- \`reports/*_confusion_matrix.png\`: visual comparison of true and predicted labels.

Do not rename a checkpoint without updating the code that expects that filename. The \`custom_\` prefix keeps Custom checkpoints separate from the original TrashNet checkpoints.

---

## 4. Setup

These commands assume the project is at \`/home/ak/Code/clgproject\` on CachyOS and Python 3.12.8 is available through pyenv. Adjust the path if the repository is elsewhere.

\`\`\`fish
cd /home/ak/Code/clgproject
pyenv shell 3.12.8
python --version
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
\`\`\`

A dedicated virtual environment is recommended for reproducibility. The commands above use the currently selected pyenv interpreter because that is the project's established local setup.

Check Python syntax before training:

\`\`\`fish
python -m py_compile src/image_registry.py src/prepare_data.py src/train.py src/evaluate.py src/app.py
\`\`\`

---

## 5. Dataset preparation commands

Run these from the repository root.

### Prepare Custom

\`\`\`fish
python src/prepare_data.py --dataset custom
\`\`\`

This creates \`data/processed/custom/train.csv\`, \`class_names.json\`, and \`dataset_metadata.json\`. It uses all labelled images found under \`data/custom_images/train/<class>/\`.

### Prepare TrashNet

\`\`\`fish
python src/prepare_data.py --dataset trashnet
\`\`\`

This creates the standard \`data/processed/train.csv\`, \`val.csv\`, \`test.csv\`, class names, and dataset metadata. TrashNet uses an approximately 70% / 15% / 15% stratified split.

Check the class folders and labels before training. A wrong folder name becomes a wrong target label.

---

## 6. Training commands

### Train Custom models on all labelled images

These commands use all Custom images. The higher epoch counts are examples from the memorization workflow; training still may not reach 100% neural-network accuracy.

\`\`\`fish
python src/train.py --dataset custom --model mobilenetv2 --head-epochs 40 --fine-tune-epochs 30
python src/train.py --dataset custom --model efficientnetb0 --head-epochs 40 --fine-tune-epochs 30
\`\`\`

Custom checkpoints:
- \`models/custom_mobilenetv2_best.keras\`
- \`models/custom_efficientnetb0_best.keras\`

Custom metadata:
- \`models/custom_mobilenetv2_metadata.json\`
- \`models/custom_efficientnetb0_metadata.json\`

### Train TrashNet models

These commands train or replace the TrashNet checkpoints. Run them only when you intend to retrain those models.

\`\`\`fish
python src/train.py --dataset trashnet --model mobilenetv2 --head-epochs 5 --fine-tune-epochs 3
python src/train.py --dataset trashnet --model efficientnetb0 --head-epochs 5 --fine-tune-epochs 3
\`\`\`

TrashNet checkpoints:
- \`models/mobilenetv2_best.keras\`
- \`models/efficientnetb0_best.keras\`

### Training arguments

- \`--dataset\`: \`custom\` or \`trashnet\`.
- \`--model\`: \`mobilenetv2\` or \`efficientnetb0\`.
- \`--head-epochs\`: number of epochs for training the classification head.
- \`--fine-tune-epochs\`: number of epochs for fine-tuning the final backbone layers; use \`0\` to skip fine-tuning.
- \`--batch-size\`: number of images per batch; default is 16.

TensorFlow may print CUDA/GPU warnings on a CPU-only machine. If training completes and saves the model, these warnings alone do not indicate training failure.

---

## 7. Evaluation commands

### Evaluate Custom

\`\`\`fish
python src/evaluate.py --dataset custom --model mobilenetv2
python src/evaluate.py --dataset custom --model efficientnetb0
\`\`\`

Custom evaluation uses \`data/processed/custom/train.csv\`, so its accuracy is **training-set memorization accuracy**.

### Evaluate TrashNet

\`\`\`fish
python src/evaluate.py --dataset trashnet --model mobilenetv2
python src/evaluate.py --dataset trashnet --model efficientnetb0
\`\`\`

TrashNet evaluation uses \`data/processed/test.csv\`, which is held out from training.

Reports are saved in \`reports/\`. Custom reports have the \`custom_\` prefix. The exact registry self-check is a separate check of image-to-folder mapping and must not be reported as neural-network accuracy.

---

## 8. Launch and test the GUI

In **Terminal 1**, start the app:

\`\`\`fish
cd /home/ak/Code/clgproject
pyenv shell 3.12.8
python src/app.py
\`\`\`

Gradio normally prints a local URL such as \`http://127.0.0.1:7860\`. Keep this terminal running while using the app.

In the GUI:
1. Select **TrashNet** or **Custom**.
2. Select **MobileNetV2** or **EfficientNetB0**.
3. Optionally enable **Compare both models**.
4. Upload one image or multiple image files.
5. Click **Classify image(s)**.
6. Inspect the final result, model predictions, model scores, top predictions, and inference latency.
7. For Custom, check the strict registry result separately from the neural-network predictions.

After retraining, stop the old GUI with \`Ctrl+C\` and launch \`python src/app.py\` again so the process reloads the updated model checkpoints.

---

## 9. Cloudflare Tunnel commands for a temporary public URL

A temporary Cloudflare Quick Tunnel does not require configuring a named tunnel or adding a domain. Install \`cloudflared\` first if it is not already available.

**Keep the Gradio application running in Terminal 1.** Open **Terminal 2** and run:

\`\`\`fish
cloudflared tunnel --url http://127.0.0.1:7860
\`\`\`

Wait for \`cloudflared\` to print a public URL ending in \`.trycloudflare.com\`. Open that URL on the device you want to use for the demo. Leave both terminals running: one for Gradio and one for Cloudflare Tunnel.

Useful checks:

\`\`\`fish
# Confirm cloudflared is installed
cloudflared --version

# Confirm the local Gradio endpoint responds (run while the app is up)
curl -I http://127.0.0.1:7860
\`\`\`

To stop the tunnel, press \`Ctrl+C\` in Terminal 2. To stop the GUI, press \`Ctrl+C\` in Terminal 1. A new Quick Tunnel normally receives a different random URL when restarted. A stable custom hostname requires a Cloudflare-managed domain and a configured named tunnel.

**Security note:** Anyone who obtains the temporary public URL may be able to access the exposed application. Do not use the public tunnel for private images, confidential data, or an app that should be restricted to your team. Stop the tunnel when the demonstration is over.

---

## 10. Git commands

Check local changes:

\`\`\`fish
git status --short
git diff --check
\`\`\`

Pull the latest shared changes:

\`\`\`fish
git pull --ff-only origin main
\`\`\`

After reviewing your own changes, stage and commit them:

\`\`\`fish
git add README.md
git commit -m "Update project documentation and team responsibilities"
git push origin main
\`\`\`

If you changed other files, stage only the intended paths or review \`git status\` carefully before committing. Do not commit API keys, private files, or datasets that you do not have permission to redistribute.

---

## 11. Known limitations

- The classifier predicts one label for the whole image; it is not an object-detection system for multiple objects in a scene.
- A model's confidence/score is not a guarantee that its prediction is correct.
- Custom accuracy is measured on the same images used for training, so it does not demonstrate generalization to unseen or modified images.
- Custom currently has no labelled \`trash\` examples. Its unmatched-image fallback is a rule, not a learned class prediction.
- Exact image matching depends on decoded pixels. Resizing, cropping, recompressing, or editing an image may prevent a match.
- CPU inference and training may be slower than GPU-backed execution.
- The temporary Cloudflare URL is intended for demonstrations, not permanent production hosting.

## 12. Dataset acknowledgement

The project uses the [TrashNet dataset](https://github.com/garythung/trashnet). Cite the dataset in reports and review its reuse terms. Do not redistribute downloaded or third-party Custom images unless the applicable license or permission allows it.
