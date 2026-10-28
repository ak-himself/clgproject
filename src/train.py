"""Train a transfer-learning waste classifier on TrashNet or RealWaste."""
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
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42


def set_seeds() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    # Keep the laptop responsive and avoid thermal saturation.
    try:
        tf.config.threading.set_intra_op_parallelism_threads(4)
        tf.config.threading.set_inter_op_parallelism_threads(2)
    except RuntimeError:
        pass


def paths_for_dataset(dataset: str) -> tuple[Path, str]:
    data_dir = PROCESSED_DIR if dataset == "trashnet" else PROCESSED_DIR / dataset
    prefix = "" if dataset == "trashnet" else f"{dataset}_"
    return data_dir, prefix


def load_manifest(data_dir: Path, name: str, class_to_index: dict[str, int]) -> pd.DataFrame:
    path = data_dir / f"{name}.csv"
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run: python src/prepare_data.py --dataset <name>")
    frame = pd.read_csv(path)
    frame["label_id"] = frame["label"].map(class_to_index)
    if frame["label_id"].isna().any():
        raise SystemExit(f"Unknown labels found in {path}")
    frame["absolute_path"] = frame["path"].map(lambda p: str(ROOT / p))
    missing = [p for p in frame["absolute_path"] if not Path(p).is_file()]
    if missing:
        raise SystemExit(f"Image file is missing, e.g. {missing[0]}")
    return frame


def make_dataset(frame: pd.DataFrame, training: bool, batch_size: int) -> tf.data.Dataset:
    paths = frame["absolute_path"].to_numpy()
    labels = frame["label_id"].astype("int32").to_numpy()
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def load_image(path: tf.Tensor, label: tf.Tensor):
        image_bytes = tf.io.read_file(path)
        image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, IMAGE_SIZE, antialias=True)
        return tf.cast(image, tf.float32), label

    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(len(frame), seed=SEED, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_model(
    model_name: str, num_classes: int
) -> tuple[keras.Model, keras.Model]:
    inputs = keras.Input(shape=(*IMAGE_SIZE, 3), name="image")
    # Augmentation is active only during training; validation, evaluation and
    # GUI inference use the unmodified image. Mild transforms preserve labels.
    augmenter = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.06),
            layers.RandomZoom(0.10),
            layers.RandomContrast(0.10),
        ],
        name="training_augmentation",
    )
    x = augmenter(inputs)

    if model_name == "mobilenetv2":
        backbone = keras.applications.MobileNetV2(
            include_top=False, weights="imagenet", input_shape=(*IMAGE_SIZE, 3)
        )
        x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="mobilenet_preprocess")(x)
    elif model_name == "efficientnetb0":
        backbone = keras.applications.EfficientNetB0(
            include_top=False, weights="imagenet", input_shape=(*IMAGE_SIZE, 3)
        )
        # EfficientNetB0 includes its own rescaling/preprocessing layers.
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    backbone.trainable = False
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.30)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    return keras.Model(inputs, outputs, name=f"{model_name}_waste_classifier"), backbone


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["trashnet", "realwaste"], default="trashnet")
    parser.add_argument("--model", choices=["mobilenetv2", "efficientnetb0"], required=True)
    parser.add_argument("--head-epochs", type=int, default=5)
    parser.add_argument("--fine-tune-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    if args.head_epochs < 1 or args.fine_tune_epochs < 0 or args.batch_size < 1:
        parser.error("epochs and batch size must be positive (fine-tune epochs may be zero)")

    set_seeds()
    data_dir, prefix = paths_for_dataset(args.dataset)
    class_file = data_dir / "class_names.json"
    if not class_file.exists():
        raise SystemExit(
            f"Class names not found at {class_file}. "
            f"Run: python src/prepare_data.py --dataset {args.dataset}"
        )
    class_names = json.loads(class_file.read_text(encoding="utf-8"))
    class_to_index = {name: index for index, name in enumerate(class_names)}
    train_frame = load_manifest(data_dir, "train", class_to_index)
    val_frame = load_manifest(data_dir, "val", class_to_index)
    train_ds = make_dataset(train_frame, training=True, batch_size=args.batch_size)
    val_ds = make_dataset(val_frame, training=False, batch_size=args.batch_size)

    model, backbone = build_model(args.model, len(class_names))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = MODEL_DIR / f"{prefix}{args.model}_best.keras"
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            checkpoint, monitor="val_loss", save_best_only=True, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=2, restore_best_weights=True
        ),
    ]
    class_ids = np.arange(len(class_names))
    weights = compute_class_weight(
        class_weight="balanced",
        classes=class_ids,
        y=train_frame["label_id"].to_numpy(dtype=int),
    )
    class_weights = {int(class_id): float(weight) for class_id, weight in zip(class_ids, weights)}

    print(f"Dataset: {args.dataset} | model: {args.model} | CPU training")
    print(f"Classes: {class_names}")
    print(f"Train/validation images: {len(train_frame)}/{len(val_frame)}")
    print(f"Using mild augmentation and class weights: {class_weights}")
    model.fit(
        train_ds, validation_data=val_ds, epochs=args.head_epochs,
        callbacks=callbacks, class_weight=class_weights, verbose=1,
    )

    if args.fine_tune_epochs > 0:
        print("Fine-tuning the last backbone layers...")
        backbone.trainable = True
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
            callbacks=callbacks, class_weight=class_weights, verbose=1,
        )

    # Reload the best validation-loss checkpoint rather than saving the final epoch.
    keras.models.load_model(checkpoint)
    metadata = {
        "dataset": args.dataset,
        "model": args.model,
        "class_names": class_names,
        "image_size": list(IMAGE_SIZE),
        "weights": str(checkpoint.relative_to(ROOT)),
        "seed": SEED,
        "augmentation": ["horizontal_flip", "rotation", "zoom", "contrast"],
    }
    (MODEL_DIR / f"{prefix}{args.model}_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nSaved best model: {checkpoint}")
    print(f"Metadata: {MODEL_DIR / f'{prefix}{args.model}_metadata.json'}")


if __name__ == "__main__":
    main()
