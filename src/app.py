"""Gradio GUI: upload a waste image and test a trained model."""
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
DATA_DIR = ROOT / "data" / "processed"
IMAGE_SIZE = (224, 224)
MODEL_NAMES = {
    "MobileNetV2": "mobilenetv2",
    "EfficientNetB0": "efficientnetb0",
}
_loaded_models: dict[str, tuple[tf.keras.Model, list[str]]] = {}


def available_models() -> list[str]:
    return [
        display for display, key in MODEL_NAMES.items()
        if (MODEL_DIR / f"{key}_best.keras").exists()
    ]


def load_model(display_name: str) -> tuple[tf.keras.Model, list[str]]:
    key = MODEL_NAMES[display_name]
    if key not in _loaded_models:
        model_path = MODEL_DIR / f"{key}_best.keras"
        class_path = DATA_DIR / "class_names.json"
        if not model_path.exists():
            raise FileNotFoundError(
                f"{model_path.name} not found. Train this model first using src/train.py."
            )
        if not class_path.exists():
            raise FileNotFoundError("Class names not found. Run src/prepare_data.py first.")
        class_names = json.loads(class_path.read_text(encoding="utf-8"))
        _loaded_models[key] = (tf.keras.models.load_model(model_path), class_names)
    return _loaded_models[key]


def predict(image: Image.Image | None, model_choice: str):
    if image is None:
        return {}, "Upload an image to begin."
    if not model_choice:
        return {}, "No trained model found. Train MobileNetV2 or EfficientNetB0 first."
    model, class_names = load_model(model_choice)
    image = image.convert("RGB").resize(IMAGE_SIZE)
    array = np.asarray(image, dtype=np.float32)[None, ...]
    probabilities = model.predict(array, verbose=0)[0]
    ranking = np.argsort(probabilities)[::-1]
    scores = {class_names[i]: float(probabilities[i]) for i in ranking}
    best = int(ranking[0])
    return scores, f"Prediction: {class_names[best]} (confidence {probabilities[best]:.1%})"


with gr.Blocks(title="Waste Classification") as demo:
    gr.Markdown(
        "# Waste Classification\n"
        "Upload an image to test a model fine-tuned on TrashNet. "
        "Confidence is the model's predicted probability, not a guarantee of correctness."
    )
    model_dropdown = gr.Dropdown(
        choices=available_models(), value=(available_models()[0] if available_models() else None),
        label="Trained model", interactive=True,
    )
    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload waste image")
        prediction_output = gr.Label(label="Class probabilities", num_top_classes=6)
    status_output = gr.Textbox(label="Result", interactive=False)
    predict_button = gr.Button("Classify image", variant="primary")
    predict_button.click(
        fn=predict, inputs=[image_input, model_dropdown],
        outputs=[prediction_output, status_output],
    )
    gr.Markdown(
        "To train a model, run the commands in README.md. "
        "Restart this app after training if you trained a new model."
    )


if __name__ == "__main__":
    demo.launch()
