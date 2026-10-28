"""Gradio GUI: upload a waste image and test a model trained on TrashNet or RealWaste."""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
PROCESSED_DIR = ROOT / "data" / "processed"
IMAGE_SIZE = (224, 224)
MODEL_LABELS = {
    "mobilenetv2": "MobileNetV2",
    "efficientnetb0": "EfficientNetB0",
}
DATASET_LABELS = {"trashnet": "TrashNet (6 classes)", "realwaste": "RealWaste (9 classes)"}
_loaded_models: dict[str, tuple[tf.keras.Model, list[str]]] = {}


def options() -> dict[str, tuple[str, str]]:
    found: dict[str, tuple[str, str]] = {}
    for dataset in ("trashnet", "realwaste"):
        prefix = "" if dataset == "trashnet" else f"{dataset}_"
        for model_key, model_label in MODEL_LABELS.items():
            checkpoint = MODEL_DIR / f"{prefix}{model_key}_best.keras"
            if checkpoint.exists():
                found[f"{DATASET_LABELS[dataset]} / {model_label}"] = (dataset, model_key)
    return found


def load_model(display_name: str) -> tuple[tf.keras.Model, list[str]]:
    dataset, model_key = options()[display_name]
    cache_key = f"{dataset}_{model_key}"
    prefix = "" if dataset == "trashnet" else f"{dataset}_"
    if cache_key not in _loaded_models:
        model_path = MODEL_DIR / f"{prefix}{model_key}_best.keras"
        data_dir = PROCESSED_DIR if dataset == "trashnet" else PROCESSED_DIR / dataset
        class_path = data_dir / "class_names.json"
        if not class_path.exists():
            raise FileNotFoundError(f"Class names not found: {class_path}")
        class_names = json.loads(class_path.read_text(encoding="utf-8"))
        _loaded_models[cache_key] = (tf.keras.models.load_model(model_path), class_names)
    return _loaded_models[cache_key]


def predict(image: Image.Image | None, model_choice: str):
    if image is None:
        return {}, "Upload an image to begin."
    available = options()
    if not model_choice or model_choice not in available:
        return {}, "No matching trained model is available. Train a model first."
    model, class_names = load_model(model_choice)
    image = image.convert("RGB").resize(IMAGE_SIZE)
    array = np.asarray(image, dtype=np.float32)[None, ...]
    probabilities = model.predict(array, verbose=0)[0]
    ranking = np.argsort(probabilities)[::-1]
    scores = {class_names[i]: float(probabilities[i]) for i in ranking}
    best = int(ranking[0])
    note = (
        f"Prediction: {class_names[best]} (model score {probabilities[best]:.1%}). "
        "The model always selects among its trained classes; an unfamiliar image can still receive a high score."
    )
    return scores, note


with gr.Blocks(title="Waste Classification Lab") as demo:
    gr.Markdown(
        "# Waste Classification Lab\n"
        "Compare models trained on the controlled TrashNet dataset and the more realistic RealWaste dataset. "
        "Scores are model probabilities, not guarantees of correctness."
    )
    model_options = options()
    model_dropdown = gr.Dropdown(
        choices=list(model_options),
        value=(next(iter(model_options)) if model_options else None),
        label="Dataset and model",
        interactive=True,
    )
    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload waste image")
        prediction_output = gr.Label(label="Class scores", num_top_classes=9)
    status_output = gr.Textbox(label="Result", interactive=False)
    predict_button = gr.Button("Classify image", variant="primary")
    predict_button.click(
        fn=predict, inputs=[image_input, model_dropdown],
        outputs=[prediction_output, status_output],
    )
    gr.Markdown(
        "For a fair external test, use images that were not included in training or validation. "
        "A single-image classifier is not designed to detect and separately label multiple objects in one photograph."
    )


if __name__ == "__main__":
    demo.launch()
