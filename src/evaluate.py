"""Evaluate a saved model on the held-out test split."""
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
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["mobilenetv2", "efficientnetb0"], required=True)
    args = parser.parse_args()

    model_path = MODEL_DIR / f"{args.model}_best.keras"
    class_path = DATA_DIR / "class_names.json"
    test_path = DATA_DIR / "test.csv"
    for path in (model_path, class_path, test_path):
        if not path.exists():
            raise SystemExit(f"Required file missing: {path}")

    class_names = json.loads(class_path.read_text(encoding="utf-8"))
    frame = pd.read_csv(test_path)
    class_to_index = {name: i for i, name in enumerate(class_names)}
    y_true = frame["label"].map(class_to_index).to_numpy(dtype=int)
    model = tf.keras.models.load_model(model_path)

    images = []
    for relative_path in frame["path"]:
        full_path = ROOT / relative_path
        image = tf.keras.utils.load_img(full_path, target_size=IMAGE_SIZE)
        images.append(tf.keras.utils.img_to_array(image))
    x_test = np.asarray(images, dtype=np.float32)

    start = time.perf_counter()
    probabilities = model.predict(x_test, batch_size=BATCH_SIZE, verbose=0)
    elapsed = time.perf_counter() - start
    y_pred = probabilities.argmax(axis=1)

    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    report = classification_report(
        y_true, y_pred, labels=list(range(len(class_names))),
        target_names=class_names, output_dict=True, zero_division=0,
    )
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "model": args.model,
        "test_samples": len(frame),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "mean_inference_ms_per_image": elapsed * 1000 / max(len(frame), 1),
        "classification_report": report,
    }
    report_path = REPORT_DIR / f"{args.model}_metrics.json"
    report_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix, interpolation="nearest")
    fig.colorbar(image, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)), yticks=np.arange(len(class_names)),
        xticklabels=class_names, yticklabels=class_names,
        xlabel="Predicted label", ylabel="True label",
        title=f"{args.model} — held-out test confusion matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center",
                    color="white" if matrix[row, col] > threshold else "black")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / f"{args.model}_confusion_matrix.png", dpi=150)
    plt.close(fig)

    print(f"Model: {args.model}")
    print(f"Test samples: {len(frame)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Mean inference: {output['mean_inference_ms_per_image']:.2f} ms/image")
    print(f"Saved report: {report_path}")
    print(f"Saved confusion matrix: {REPORT_DIR / f'{args.model}_confusion_matrix.png'}")


if __name__ == "__main__":
    main()
