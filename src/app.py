"""Gradio GUI for the trained TrashNet classifiers."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import gradio as gr
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MODEL_LABELS = {
    "mobilenetv2": "MobileNetV2",
    "efficientnetb0": "EfficientNetB0",
}
MODEL_KEYS = ("efficientnetb0", "mobilenetv2")


@dataclass(frozen=True)
class ModelSpec:
    model_key: str
    display_name: str
    model: tf.keras.Model | None
    class_names: tuple[str, ...]
    image_size: tuple[int, int]
    error: str | None = None


def _paths_for(model_key: str) -> tuple[Path, Path]:
    return (
        MODEL_DIR / f"{model_key}_best.keras",
        MODEL_DIR / f"{model_key}_metadata.json",
    )


def _display_name(model_key: str) -> str:
    return f"TrashNet / {MODEL_LABELS[model_key]}"


def _failed_spec(model_key: str, error: str) -> ModelSpec:
    return ModelSpec(
        model_key=model_key,
        display_name=_display_name(model_key),
        model=None,
        class_names=(),
        image_size=(224, 224),
        error=error,
    )


def _load_spec(model_key: str) -> ModelSpec:
    model_path, metadata_path = _paths_for(model_key)
    display_name = _display_name(model_key)
    if not model_path.is_file():
        return _failed_spec(model_key, f"Model file is missing: {model_path}")
    if not metadata_path.is_file():
        return _failed_spec(model_key, f"Metadata file is missing: {metadata_path}")

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        class_names = tuple(metadata["class_names"])
        image_size = tuple(metadata["image_size"])
        if len(class_names) == 0 or len(image_size) != 2:
            raise ValueError("metadata has an invalid class_names or image_size value")
        model = tf.keras.models.load_model(model_path)
    except (OSError, ValueError, KeyError, TypeError, tf.errors.OpError) as exc:
        return _failed_spec(model_key, f"Could not load {display_name}: {exc}")

    return ModelSpec(
        model_key=model_key,
        display_name=display_name,
        model=model,
        class_names=class_names,
        image_size=(int(image_size[0]), int(image_size[1])),
    )


def load_models() -> dict[str, ModelSpec]:
    """Load every known checkpoint once when the application starts."""
    return {
        model_key: _load_spec(model_key) for model_key in MODEL_KEYS
    }


_MODELS = load_models()


def model_choices() -> list[str]:
    return [_MODELS[model_key].display_name for model_key in MODEL_KEYS]


def _selected_spec(model_name: str) -> ModelSpec:
    for model_key in MODEL_KEYS:
        spec = _MODELS[model_key]
        if spec.display_name == model_name:
            return spec
    raise ValueError("The selected model is not available.")


def _prepare_image(image: Image.Image, spec: ModelSpec) -> np.ndarray:
    if not isinstance(image, Image.Image):
        raise ValueError("The uploaded file is not a valid image.")
    resized = image.convert("RGB").resize(spec.image_size)
    return np.asarray(resized, dtype=np.float32)[None, ...]


def predict_image(image: Image.Image, spec: ModelSpec) -> dict[str, Any]:
    if spec.model is None:
        raise RuntimeError(spec.error or "The selected model could not be loaded.")
    array = _prepare_image(image, spec)
    started = time.perf_counter()
    probabilities = np.asarray(spec.model.predict(array, verbose=0))[0]
    latency_ms = (time.perf_counter() - started) * 1000
    if probabilities.ndim != 1 or len(probabilities) != len(spec.class_names):
        raise RuntimeError(
            f"{spec.display_name} returned {len(probabilities)} scores for "
            f"{len(spec.class_names)} classes."
        )
    ranking = np.argsort(probabilities)[::-1]
    top = [
        (spec.class_names[int(index)], float(probabilities[int(index)]))
        for index in ranking[:3]
    ]
    return {
        "prediction": top[0][0],
        "score": top[0][1],
        "top": top,
        "latency_ms": latency_ms,
    }


def _format_result(result: dict[str, Any], model_name: str) -> str:
    top_lines = "\n".join(
        f"{index}. **{label}** — model score **{score:.1%}**"
        for index, (label, score) in enumerate(result["top"], start=1)
    )
    return (
        f"### {model_name}\n"
        f"**Prediction:** {result['prediction']}  \n"
        f"**Inference latency:** {result['latency_ms']:.1f} ms  \n\n"
        f"**Top 3 predictions**\n{top_lines}"
    )


def _file_path(file_value: Any) -> str:
    if isinstance(file_value, str):
        return file_value
    path = getattr(file_value, "path", None)
    if path:
        return str(path)
    name = getattr(file_value, "name", None)
    if name:
        return str(name)
    raise ValueError("An uploaded file has no readable path.")


def classify(
    image: Image.Image | None,
    batch_files: list[Any] | None,
    model_name: str,
    compare: bool,
):
    status_messages: list[str] = []
    selected_spec = None
    try:
        selected_spec = _selected_spec(model_name)
    except ValueError as exc:
        status_messages.append(str(exc))

    specs = [selected_spec] if selected_spec else []
    if compare and selected_spec:
        comparison_model = (
            "mobilenetv2"
            if selected_spec and selected_spec.model_key == "efficientnetb0"
            else "efficientnetb0"
        )
        comparison_spec = _MODELS[comparison_model]
        if comparison_spec.model is None:
            status_messages.append(comparison_spec.error or "Comparison model is unavailable.")
        elif comparison_spec not in specs:
            specs.append(comparison_spec)

    result_markdown: list[str] = []
    preview = image
    rows: list[dict[str, Any]] = []
    if image is not None:
        for spec in specs:
            try:
                result = predict_image(image, spec)
                result_markdown.append(_format_result(result, spec.display_name))
            except (OSError, ValueError, RuntimeError, tf.errors.OpError) as exc:
                status_messages.append(f"{spec.display_name}: {exc}")

    for file_value in batch_files or []:
        try:
            path = _file_path(file_value)
            with Image.open(path) as opened:
                batch_image = opened.convert("RGB")
                if preview is None:
                    preview = batch_image.copy()
                for spec in specs:
                    result = predict_image(batch_image, spec)
                    rows.append(
                        {
                            "Image": Path(path).name,
                            "Model": spec.display_name,
                            "Prediction": result["prediction"],
                            "Model score": f"{result['score']:.1%}",
                            "Top 3": " | ".join(
                                f"{label} ({score:.1%})"
                                for label, score in result["top"]
                            ),
                            "Latency (ms)": round(result["latency_ms"], 1),
                        }
                    )
        except (OSError, ValueError, RuntimeError, tf.errors.OpError) as exc:
            status_messages.append(f"{getattr(file_value, 'name', 'Uploaded file')}: {exc}")

    if not image and not batch_files:
        status_messages.append("Upload an image or one or more files to begin.")
    status_messages.append(
        "Model scores are not guarantees of correctness. The classifier always selects a known class."
    )
    return (
        preview,
        "\n\n".join(result_markdown) or "No single-image result yet.",
        pd.DataFrame(
            rows,
            columns=["Image", "Model", "Prediction", "Model score", "Top 3", "Latency (ms)"],
        ),
        "\n".join(status_messages),
    )


with gr.Blocks(title="Waste Classification Lab") as demo:
    gr.Markdown(
        "# Waste Classification Lab\n"
        "Upload one image for a detailed prediction or upload multiple files for a batch table. "
        "Use Compare both models to run both TrashNet backbones on the same images."
    )
    with gr.Row():
        model_dropdown = gr.Dropdown(
            choices=model_choices(),
            value=_MODELS["efficientnetb0"].display_name,
            label="Model",
        )
        compare_checkbox = gr.Checkbox(label="Compare both models", value=False)
    with gr.Row():
        image_input = gr.Image(type="pil", label="Single image")
        preview_output = gr.Image(type="pil", label="Uploaded image", interactive=False)
    batch_input = gr.File(
        file_count="multiple", file_types=["image"], type="filepath", label="Batch images"
    )
    classify_button = gr.Button("Classify image(s)", variant="primary")
    single_output = gr.Markdown("No single-image result yet.")
    batch_output = gr.Dataframe(
        headers=["Image", "Model", "Prediction", "Model score", "Top 3", "Latency (ms)"],
        datatype=["str", "str", "str", "str", "str", "number"],
        label="Batch results",
        interactive=False,
    )
    status_output = gr.Textbox(label="Status", interactive=False)

    classify_button.click(
        fn=classify,
        inputs=[image_input, batch_input, model_dropdown, compare_checkbox],
        outputs=[preview_output, single_output, batch_output, status_output],
    )
    gr.Markdown(
        "Confidence is shown as a model score, not a guarantee of correctness. "
        "This closed-set classifier labels the whole image and does not detect separate objects."
    )


if __name__ == "__main__":
    demo.launch()
