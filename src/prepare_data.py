"""Create reproducible train/validation/test CSV manifests from TrashNet.

Expected source: data/raw/dataset-resized/<class_name>/*.{jpg,jpeg,png}
No image files are copied; the manifests reference the original images.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "raw" / "dataset-resized"
OUTPUT = ROOT / "data" / "processed"
SEED = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    if not SOURCE.is_dir():
        raise SystemExit(
            f"Dataset directory not found: {SOURCE}\n"
            "Extract TrashNet so that data/raw/dataset-resized/<class>/*.jpg exists."
        )

    rows: list[dict[str, str]] = []
    class_dirs = sorted(
        p for p in SOURCE.iterdir()
        if p.is_dir() and not p.name.startswith(".") and p.name != "__MACOSX"
    )
    for class_dir in class_dirs:
        for image_path in sorted(class_dir.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                rows.append({
                    "path": image_path.relative_to(ROOT).as_posix(),
                    "label": class_dir.name,
                })

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise SystemExit(f"No images found under {SOURCE}")
    counts = frame["label"].value_counts()
    if len(counts) < 2 or counts.min() < 3:
        raise SystemExit("Each class must contain at least 3 images for stratified splitting.")

    train_val, test = train_test_split(
        frame, test_size=TEST_SIZE, random_state=SEED, stratify=frame["label"]
    )
    # 15% of all data for validation => 15/85 of the remaining train_val.
    train, val = train_test_split(
        train_val,
        test_size=VAL_SIZE / (1.0 - TEST_SIZE),
        random_state=SEED,
        stratify=train_val["label"],
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, split in (("train", train), ("val", val), ("test", test)):
        split.sort_values(["label", "path"]).to_csv(OUTPUT / f"{name}.csv", index=False)
        print(f"{name:5s}: {len(split):4d} images | {split['label'].nunique()} classes")

    class_names = sorted(frame["label"].unique().tolist())
    (OUTPUT / "class_names.json").write_text(
        json.dumps(class_names, indent=2) + "\n", encoding="utf-8"
    )
    print("\nClass counts:")
    print(frame["label"].value_counts().sort_index().to_string())
    print(f"\nSaved manifests to: {OUTPUT}")
    print("Split proportions are approximately 70% train / 15% validation / 15% test.")


if __name__ == "__main__":
    main()
