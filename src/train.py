"""Train a transfer-learning waste classifier on the prepared TrashNet manifests."""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42


def set_seeds() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    try:
        tf.config.threading.set_intra_op_parallelism_threads(4)
        tf.config.threading.set_inter_op_parallelism_threads(2)
    except RuntimeError:
        pass


def load_manifest(name: str, class_to_index: dict[str, int]) -> pd.DataFrame:
    path = DATA_DIR / f"{name}.csv"
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run: python src/prepare_data.py")
    frame = pd.read_csv(path)
    frame["label_id"] = frame["label"].map(class_to_index)
    if frame["label_id"].isna().any():
        raise SystemExit(f"Unknown labels found in {path}")
    frame["absolute_path"] = frame["path"].map(lambda p: str(ROOT / p))
    return frame


def make_dataset(frame: pd.DataFrame, training: bool) -> tf.data.Dataset:
    paths = frame["absolute_path"].to_numpy()
    labels = frame["label_id"].astype("int32").to_numpy()
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def load_image(path: tf.Tensor, label: tf.Tensor):
        image_bytes = tf.io.read_file(path)
        image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, IMAGE_SIZE, antialias=True)
        # Keep values in [0, 255]. Preprocessing is part of each model.
        image = tf.cast(image, tf.float32)
        return image, label

    ds = ds.map(load_image, num_parallel_calls=2)
    if training:
        ds = ds.shuffle(len(frame), seed=SEED, reshuffle_each_iteration=True)
    return ds.batch(BATCH_SIZE).prefetch(1)


def build_model(model_name: str, num_classes: int) -> tuple[keras.Model, keras.Model]:
    inputs = keras.Input(shape=(*IMAGE_SIZE, 3), name="image")
    if model_name == "mobilenetv2":
        backbone = keras.applications.MobileNetV2(
            include_top=False, weights="imagenet", input_shape=(*IMAGE_SIZE, 3)
        )
        x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="mobilenet_preprocess")(inputs)
    elif model_name == "efficientnetb0":
        backbone = keras.applications.EfficientNetB0(
            include_top=False, weights="imagenet", input_shape=(*IMAGE_SIZE, 3)
        )
        # EfficientNetB0 includes its own rescaling/preprocessing layers.
        x = inputs
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    backbone.trainable = False
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    return keras.Model(inputs, outputs, name=f"trash_{model_name}"), backbone


def main() -> None:
    global BATCH_SIZE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["mobilenetv2", "efficientnetb0"], required=True)
    parser.add_argument("--head-epochs", type=int, default=5)
    parser.add_argument("--fine-tune-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    BATCH_SIZE = args.batch_size
    set_seeds()

    class_file = DATA_DIR / "class_names.json"
    if not class_file.exists():
        raise SystemExit("Class names not found. Run: python src/prepare_data.py")
    class_names = json.loads(class_file.read_text(encoding="utf-8"))
    class_to_index = {name: index for index, name in enumerate(class_names)}
    train_frame = load_manifest("train", class_to_index)
    val_frame = load_manifest("val", class_to_index)
    train_ds = make_dataset(train_frame, training=True)
    val_ds = make_dataset(val_frame, training=False)

    model, backbone = build_model(args.model, len(class_names))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = MODEL_DIR / f"{args.model}_best.keras"
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            checkpoint, monitor="val_loss", save_best_only=True, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=2, restore_best_weights=True
        ),
    ]

    print(f"Training {args.model} on CPU | classes={class_names}")
    model.fit(
        train_ds, validation_data=val_ds, epochs=args.head_epochs,
        callbacks=callbacks, verbose=1,
    )

    if args.fine_tune_epochs > 0:
        print("Fine-tuning the last backbone layers...")
        backbone.trainable = True
        # Keep most layers frozen to reduce CPU cost; keep BatchNorm frozen for stability.
        for layer in backbone.layers[:-30]:
            layer.trainable = False
        for layer in backbone.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-5),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.fit(
            train_ds, validation_data=val_ds, epochs=args.fine_tune_epochs,
            callbacks=callbacks, verbose=1,
        )

    best_model = keras.models.load_model(checkpoint)
    best_model.save(checkpoint)
    metadata = {
        "model": args.model,
        "class_names": class_names,
        "image_size": list(IMAGE_SIZE),
        "weights": str(checkpoint.relative_to(ROOT)),
        "seed": SEED,
    }
    (MODEL_DIR / f"{args.model}_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nSaved best model: {checkpoint}")
    print(f"Metadata: {MODEL_DIR / f'{args.model}_metadata.json'}")


if __name__ == "__main__":
    main()
