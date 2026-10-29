"""Gradio GUI for separate TrashNet and Custom waste classifiers."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
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

from image_registry import build_custom_registry, canonical_image_hash

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
DATASET_LABELS = {"trashnet": "TrashNet", "custom": "Custom"}
MODEL_LABELS = {"mobilenetv2": "MobileNetV2", "efficientnetb0": "EfficientNetB0"}
MODEL_KEYS = ("efficientnetb0", "mobilenetv2")
DATASET_KEYS = ("trashnet", "custom")


@dataclass(frozen=True)
class ModelSpec:
    dataset_key: str
    model_key: str
    display_name: str
    model: tf.keras.Model | None
    class_names: tuple[str, ...]
    image_size: tuple[int, int]
    error: str | None = None


def _paths_for(dataset_key: str, model_key: str) -> tuple[Path, Path]:
    prefix = "" if dataset_key == "trashnet" else "custom_"
    return (
        MODEL_DIR / f"{prefix}{model_key}_best.keras",
        MODEL_DIR / f"{prefix}{model_key}_metadata.json",
    )


def _display_name(dataset_key: str, model_key: str) -> str:
    return f"{DATASET_LABELS[dataset_key]} / {MODEL_LABELS[model_key]}"


def _failed_spec(dataset_key: str, model_key: str, error: str) -> ModelSpec:
    return ModelSpec(
        dataset_key=dataset_key,
        model_key=model_key,
        display_name=_display_name(dataset_key, model_key),
        model=None,
        class_names=(),
        image_size=(224, 224),
        error=error,
    )


def _load_spec(dataset_key: str, model_key: str) -> ModelSpec:
    model_path, metadata_path = _paths_for(dataset_key, model_key)
    display_name = _display_name(dataset_key, model_key)
    if not model_path.is_file():
        return _failed_spec(dataset_key, model_key, f"Model file is missing: {model_path}")
    if not metadata_path.is_file():
        return _failed_spec(dataset_key, model_key, f"Metadata file is missing: {metadata_path}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("dataset") != dataset_key or metadata.get("model") != model_key:
            raise ValueError("checkpoint metadata does not match the selected dataset/model")
        class_names = tuple(metadata["class_names"])
        image_size = tuple(metadata["image_size"])
        if not class_names or len(image_size) != 2:
            raise ValueError("metadata has an invalid class_names or image_size value")
        model = tf.keras.models.load_model(model_path)
    except (OSError, ValueError, KeyError, TypeError, tf.errors.OpError) as exc:
        return _failed_spec(dataset_key, model_key, f"Could not load {display_name}: {exc}")
    return ModelSpec(
        dataset_key=dataset_key,
        model_key=model_key,
        display_name=display_name,
        model=model,
        class_names=class_names,
        image_size=(int(image_size[0]), int(image_size[1])),
    )


def load_models() -> dict[tuple[str, str], ModelSpec]:
    return {
        (dataset_key, model_key): _load_spec(dataset_key, model_key)
        for dataset_key in DATASET_KEYS
        for model_key in MODEL_KEYS
    }


_MODELS = load_models()
_CUSTOM_REGISTRY = build_custom_registry()


def _selected_spec(dataset_key: str, model_key: str) -> ModelSpec:
    if dataset_key not in DATASET_KEYS or model_key not in MODEL_KEYS:
        raise ValueError("Select a valid dataset and model.")
    return _MODELS[(dataset_key, model_key)]


def _prepare_image(image: Image.Image, spec: ModelSpec) -> np.ndarray:
    if not isinstance(image, Image.Image):
        raise ValueError("The uploaded file is not a valid image.")
    resized = image.convert("RGB").resize((spec.image_size[1], spec.image_size[0]))
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


def _format_result(result: dict[str, Any], display_name: str) -> str:
    top_lines = "\n".join(
        f"{index}. **{label}** — model score **{score:.1%}**"
        for index, (label, score) in enumerate(result["top"], start=1)
    )
    return (
        f"#### {display_name} model prediction: **{result['prediction']}**\n\n"
        f"Model score: **{result['score']:.1%}** · "
        f"Inference: **{result['latency_ms']:.1f} ms**\n\n"
        f"Top predictions:\n{top_lines}"
    )


def _write_tunnel_qr(url: str) -> None:
    """Save a QR code for the public tunnel URL in the project directory."""
    try:
        import qrcode
    except ImportError:
        print(
            "QR package is missing; install it with: "
            "python -m pip install 'qrcode[pil]'",
            flush=True,
        )
        return

    qr_path = ROOT / "tunnel_qr.png"
    try:
        import textwrap
        from PIL import Image, ImageDraw, ImageFont

        qr_image = qrcode.make(url).convert("RGB")
        font = ImageFont.load_default()
        lines = textwrap.wrap(url, width=52, break_long_words=True, break_on_hyphens=False)
        label_lines = ["Open this URL:", *lines]
        line_height = 22
        side_padding = 24
        canvas_width = max(qr_image.width + 2 * side_padding, 560)
        top_padding = 20
        gap = 18
        bottom_padding = 20
        text_height = len(label_lines) * line_height
        canvas_height = top_padding + qr_image.height + gap + text_height + bottom_padding
        canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
        qr_x = (canvas_width - qr_image.width) // 2
        canvas.paste(qr_image, (qr_x, top_padding))

        draw = ImageDraw.Draw(canvas)
        text_y = top_padding + qr_image.height + gap
        for line in label_lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            text_width = bbox[2] - bbox[0]
            text_x = max(side_padding, (canvas_width - text_width) // 2)
            draw.text((text_x, text_y), line, fill="black", font=font)
            text_y += line_height

        canvas.save(qr_path)
        print(f"QR code with visible URL saved to: {qr_path}", flush=True)
    except (OSError, ValueError) as exc:
        print(f"Could not save tunnel QR code: {exc}", flush=True)


def _start_cloudflare_tunnel(port: int) -> subprocess.Popen[str] | None:
    """Start a Cloudflare Quick Tunnel and print its URL when available."""
    executable = shutil.which("cloudflared")
    if executable is None:
        print(
            "Cloudflare Tunnel was not started because 'cloudflared' is not installed "
            "or is not on PATH. The app will remain available locally.",
            flush=True,
        )
        print("Install cloudflared, then run app.py again.", flush=True)
        return None

    try:
        process = subprocess.Popen(
            [executable, "tunnel", "--url", f"http://127.0.0.1:{port}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except OSError as exc:
        print(f"Could not start Cloudflare Tunnel: {exc}", flush=True)
        return None

    def read_tunnel_output() -> None:
        if process.stdout is None:
            return
        url_pattern = re.compile(r"https://[A-Za-z0-9-]+\.trycloudflare\.com")
        url_printed = False
        for raw_line in process.stdout:
            line = raw_line.rstrip()
            match = url_pattern.search(line)
            if match and not url_printed:
                url = match.group(0)
                print(f"\nPublic Cloudflare Tunnel URL: {url}", flush=True)
                _write_tunnel_qr(url)
                print("Anyone with this URL may be able to access the app.", flush=True)
                url_printed = True
            else:
                print(f"[cloudflared] {line}", flush=True)
        return_code = process.wait()
        if return_code not in (0, -15):
            print(f"Cloudflare Tunnel exited with code {return_code}.", flush=True)

    threading.Thread(target=read_tunnel_output, name="cloudflare-tunnel-output", daemon=True).start()
    return process


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


def _strict_custom_result(image: Image.Image) -> tuple[str, str]:
    digest = canonical_image_hash(image)
    label = _CUSTOM_REGISTRY.get(digest)
    if label is not None:
        return label, "Exact pixel match in Custom registry"
    return "trash", "Not one of the registered Custom images; strict fallback policy"


def _predict_for_specs(
    image: Image.Image, dataset_key: str, specs: list[ModelSpec]
) -> tuple[list[str], list[dict[str, Any]]]:
    messages: list[str] = []
    results: list[dict[str, Any]] = []
    if dataset_key == "custom":
        final_label, source = _strict_custom_result(image)
        messages.append(
            f"### Strict Custom result: **{final_label}**\n\n"
            f"Source: {source}. Registry entries: {len(_CUSTOM_REGISTRY)}. "
            "An unregistered image is assigned to trash by the requested policy."
        )
    else:
        final_label, source = "", ""

    for spec in specs:
        if spec.model is None:
            messages.append(f"**{spec.display_name}:** unavailable — {spec.error}")
            results.append({
                "Model": spec.display_name,
                "Model prediction": "Unavailable",
                "Model score": "",
                "Top 3": "",
                "Latency (ms)": None,
                "error": spec.error,
            })
            continue
        try:
            result = predict_image(image, spec)
            messages.append(_format_result(result, spec.display_name))
            results.append({
                "Model": spec.display_name,
                "Model prediction": result["prediction"],
                "Model score": f"{result['score']:.1%}",
                "Top 3": " | ".join(f"{label} ({score:.1%})" for label, score in result["top"]),
                "Latency (ms)": round(result["latency_ms"], 1),
                "error": None,
            })
            if dataset_key == "trashnet":
                final_label, source = result["prediction"], "ML model prediction"
        except (OSError, ValueError, RuntimeError, tf.errors.OpError) as exc:
            messages.append(f"**{spec.display_name}:** {exc}")
            results.append({
                "Model": spec.display_name,
                "Model prediction": "Error",
                "Model score": "",
                "Top 3": "",
                "Latency (ms)": None,
                "probabilities": None,
                "error": str(exc),
            })
    if dataset_key == "trashnet":
        messages.insert(0, f"### TrashNet result: **{final_label or 'Unavailable'}**\n\nSource: {source or 'No model prediction available'}")
    return messages, results


def classify(
    image: Image.Image | None,
    batch_files: list[Any] | None,
    dataset_key: str,
    model_key: str,
    compare: bool,
):
    status_messages: list[str] = []
    specs: list[ModelSpec] = []
    try:
        selected = _selected_spec(dataset_key, model_key)
        specs.append(selected)
        if compare:
            other_key = "mobilenetv2" if model_key == "efficientnetb0" else "efficientnetb0"
            specs.append(_selected_spec(dataset_key, other_key))
    except ValueError as exc:
        status_messages.append(str(exc))

    preview = image
    result_markdown: list[str] = []
    rows: list[dict[str, Any]] = []

    def add_rows(filename: str, current_image: Image.Image) -> None:
        messages, image_results = _predict_for_specs(current_image, dataset_key, specs)
        result_markdown.extend(messages)
        if dataset_key == "custom":
            final_label, source = _strict_custom_result(current_image)
        else:
            available = next((item for item in image_results if item.get("error") is None), None)
            final_label = available["Model prediction"] if available else "Unavailable"
            source = "ML model prediction"
        for item in image_results:
            rows.append({
                "Image": filename,
                "Dataset": DATASET_LABELS[dataset_key],
                "Final result": final_label,
                "Result source": source,
                **{key: value for key, value in item.items() if key != "error"},
            })

    if image is not None and specs:
        add_rows("Single image", image)

    for file_value in batch_files or []:
        try:
            path = _file_path(file_value)
            with Image.open(path) as opened:
                batch_image = opened.copy()
            if preview is None:
                preview = batch_image.copy()
            if specs:
                add_rows(Path(path).name, batch_image)
        except (OSError, ValueError, RuntimeError, tf.errors.OpError) as exc:
            status_messages.append(f"{getattr(file_value, 'name', 'Uploaded file')}: {exc}")

    if image is None and not batch_files:
        status_messages.append("Upload an image or one or more files to begin.")
    if dataset_key == "custom":
        status_messages.append(
            "Strict Custom matching guarantees the saved folder label only for exact decoded-pixel matches. "
            "Changed, resized, or cropped images are not exact matches and follow the fallback-to-trash policy. "
            "Model predictions are shown separately and are not guarantees."
        )
    else:
        status_messages.append("Model scores are not guarantees of correctness; TrashNet is a closed-set classifier.")
    table_columns = [
        "Image", "Dataset", "Final result", "Result source", "Model",
        "Model prediction", "Model score", "Top 3", "Latency (ms)",
    ]
    return (
        preview,
        "\n\n---\n\n".join(result_markdown) or "No prediction result yet.",
        pd.DataFrame(rows, columns=table_columns),
        "\n".join(status_messages),
    )


with gr.Blocks(title="Waste Classification Lab") as demo:
    gr.Markdown(
        "# Waste Classification Lab\n"
        "Choose a dataset and backbone independently. Upload a single image or a batch and inspect predictions."
    )
    with gr.Row():
        dataset_dropdown = gr.Dropdown(
            choices=[("TrashNet", "trashnet"), ("Custom", "custom")],
            value="trashnet",
            label="Dataset",
        )
        model_dropdown = gr.Dropdown(
            choices=[(MODEL_LABELS[key], key) for key in MODEL_KEYS],
            value="efficientnetb0",
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
    single_output = gr.Markdown("No prediction result yet.")
    batch_output = gr.Dataframe(
        headers=[
            "Image", "Dataset", "Final result", "Result source", "Model",
            "Model prediction", "Model score", "Top 3", "Latency (ms)",
        ],
        label="Batch results",
        interactive=False,
    )
    status_output = gr.Textbox(label="Status", interactive=False)
    classify_button.click(
        fn=classify,
        inputs=[image_input, batch_input, dataset_dropdown, model_dropdown, compare_checkbox],
        outputs=[preview_output, single_output, batch_output, status_output],
    )
    gr.Markdown(
        "Model scores are not guarantees of correctness. This is a closed-set image classifier and predicts one label for the whole image."
    )


if __name__ == "__main__":
    port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    demo.launch(
        server_name="127.0.0.1",
        server_port=port,
        prevent_thread_lock=True,
    )
    tunnel_process: subprocess.Popen[str] | None = _start_cloudflare_tunnel(port)
    print(f"Gradio is running locally at http://127.0.0.1:{port}", flush=True)
    print("Press Ctrl+C to stop the application and tunnel.", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down Gradio and Cloudflare Tunnel...", flush=True)
    finally:
        demo.close()
        if tunnel_process is not None and tunnel_process.poll() is None:
            tunnel_process.terminate()
            try:
                tunnel_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                tunnel_process.kill()
                tunnel_process.wait()
