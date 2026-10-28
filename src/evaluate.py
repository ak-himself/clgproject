"""Evaluate a model; Custom reports memorization on all training images."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from image_registry import build_custom_registry, image_file_hash

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_ROOT = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16


def processed_dir(dataset: str) -> Path:
    return PROCESSED_ROOT if dataset == "trashnet" else PROCESSED_ROOT / "custom"


def artifact_prefix(dataset: str) -> str:
    return "" if dataset == "trashnet" else "custom_"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["trashnet", "custom"], default="trashnet")
    parser.add_argument("--model", choices=["mobilenetv2", "efficientnetb0"], required=True)
    args = parser.parse_args()

    data_dir = processed_dir(args.dataset)
    prefix = artifact_prefix(args.dataset)
    model_path = MODEL_DIR / f"{prefix}{args.model}_best.keras"
    class_path = data_dir / "class_names.json"
    manifest_name = "train.csv" if args.dataset == "custom" else "test.csv"
    manifest_path = data_dir / manifest_name
    for path in (model_path, class_path, manifest_path):
        if not path.exists():
            raise SystemExit(f"Required file missing: {path}")

    class_names = json.loads(class_path.read_text(encoding="utf-8"))
    frame = pd.read_csv(manifest_path)
    if frame.empty:
        raise SystemExit(f"No images found in {manifest_path}")
    class_to_index = {name: i for i, name in enumerate(class_names)}
    y_true = frame["label"].map(class_to_index)
    if y_true.isna().any():
        raise SystemExit(f"Manifest contains labels absent from {class_path}")
    y_true = y_true.to_numpy(dtype=int)
    model = tf.keras.models.load_model(model_path)

    predictions: list[np.ndarray] = []
    inference_seconds = 0.0
    for start in range(0, len(frame), BATCH_SIZE):
        batch = frame.iloc[start : start + BATCH_SIZE]
        images = []
        for relative_path in batch["path"]:
            image = tf.keras.utils.load_img(ROOT / relative_path, target_size=IMAGE_SIZE)
            images.append(tf.keras.utils.img_to_array(image))
        x_batch = np.asarray(images, dtype=np.float32)
        tic = time.perf_counter()
        batch_probs = model.predict(x_batch, verbose=0)
        inference_seconds += time.perf_counter() - tic
        predictions.append(batch_probs)

    probabilities = np.concatenate(predictions, axis=0)
    y_pred = probabilities.argmax(axis=1)
    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    report = classification_report(
        y_true, y_pred, labels=list(range(len(class_names))),
        target_names=class_names, output_dict=True, zero_division=0,
    )
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    memorization_mode = args.dataset == "custom"
    output = {
        "dataset": args.dataset,
        "model": args.model,
        "evaluation_type": "training_set_memorization_check" if memorization_mode else "held_out_test",
        "sample_count": len(frame),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "mean_inference_ms_per_image": inference_seconds * 1000 / max(len(frame), 1),
        "classification_report": report,
    }

    if memorization_mode:
        registry = build_custom_registry()
        resolved = 0
        for row in frame.itertuples(index=False):
            try:
                matched_label = registry.get(image_file_hash(ROOT / row.path))
            except (OSError, ValueError):
                matched_label = None
            if matched_label == row.label:
                resolved += 1
        output["exact_registry_self_check"] = {
            "registered_files": len(frame),
            "files_resolved_to_expected_folder_label": resolved,
            "coverage": resolved / max(len(frame), 1),
            "note": "Registry self-check only; not model accuracy.",
        }

    report_path = REPORT_DIR / f"{prefix}{args.model}_metrics.json"
    report_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(max(8, len(class_names)), max(7, len(class_names) - 1)))
    image = ax.imshow(matrix, interpolation="nearest")
    fig.colorbar(image, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)), yticks=np.arange(len(class_names)),
        xticklabels=class_names, yticklabels=class_names,
        xlabel="Predicted label", ylabel="True label",
        title=f"{args.dataset} / {args.model} — {'training-set memorization' if memorization_mode else 'held-out test'}",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center",
                    color="white" if matrix[row, col] > threshold else "black")
    fig.tight_layout()
    matrix_path = REPORT_DIR / f"{prefix}{args.model}_confusion_matrix.png"
    fig.savefig(matrix_path, dpi=150)
    plt.close(fig)

    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    if memorization_mode:
        print(f"Training images evaluated: {len(frame)}")
        print(f"Training-set memorization accuracy: {accuracy:.4f} (not held-out test accuracy)")
    else:
        print(f"Test samples: {len(frame)}")
        print(f"Model test accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Mean model inference: {output['mean_inference_ms_per_image']:.2f} ms/image")
    if memorization_mode:
        check = output["exact_registry_self_check"]
        print(
            "Exact registry self-check: "
            f"{check['files_resolved_to_expected_folder_label']}/"
            f"{check['registered_files']} files "
            f"({check['coverage']:.1%}); this is not model accuracy."
        )
    print(f"Saved report: {report_path}")
    print(f"Saved confusion matrix: {matrix_path}")


if __name__ == "__main__":
    main()
