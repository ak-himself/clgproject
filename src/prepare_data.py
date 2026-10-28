"""Create reproducible train/validation/test manifests for TrashNet.

TrashNet: data/raw/dataset-resized/<class>/*.jpg

No images are copied. CSV manifests contain paths relative to the project root.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUTPUT_ROOT = ROOT / "data" / "processed"
SEED = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def image_files(directory: Path) -> list[Path]:
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def class_directories(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        p for p in directory.iterdir()
        if p.is_dir()
        and not p.name.startswith(".")
        and p.name != "__MACOSX"
        and image_files(p)
    )


def locate_source() -> Path:
    source = RAW_DIR / "dataset-resized"
    if source.is_dir() and len(class_directories(source)) >= 2:
        return source
    raise SystemExit(
        f"TrashNet folders not found at {source}. "
        "Extract the resized archive so each class has its own folder."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    source = locate_source()
    output = OUTPUT_ROOT

    rows: list[dict[str, str]] = []
    class_dirs = class_directories(source)
    for class_dir in class_dirs:
        label = class_dir.name.strip().lower().replace("_", " ").replace("-", " ")
        for image_path in image_files(class_dir):
            rows.append({
                "path": image_path.relative_to(ROOT).as_posix(),
                "label": label,
            })

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise SystemExit(f"No images found under {source}")
    counts = frame["label"].value_counts()
    if len(counts) < 2 or counts.min() < 3:
        raise SystemExit("Each class must contain at least 3 images for stratified splitting.")

    train_val, test = train_test_split(
        frame, test_size=TEST_SIZE, random_state=SEED, stratify=frame["label"]
    )
    # 15% of the complete dataset for validation.
    train, val = train_test_split(
        train_val,
        test_size=VAL_SIZE / (1.0 - TEST_SIZE),
        random_state=SEED,
        stratify=train_val["label"],
    )

    output.mkdir(parents=True, exist_ok=True)
    for name, split in (("train", train), ("val", val), ("test", test)):
        split.sort_values(["label", "path"]).to_csv(output / f"{name}.csv", index=False)
        print(f"{name:5s}: {len(split):4d} images | {split['label'].nunique()} classes")

    class_names = sorted(frame["label"].unique().tolist())
    (output / "class_names.json").write_text(
        json.dumps(class_names, indent=2) + "\n", encoding="utf-8"
    )
    metadata = {
        "dataset": "trashnet",
        "source_directory": source.relative_to(ROOT).as_posix(),
        "seed": SEED,
        "split": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "total_images": len(frame),
        "class_counts": {k: int(v) for k, v in counts.sort_index().items()},
        "classes": class_names,
    }
    (output / "dataset_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print("\nDataset class counts:")
    print(counts.sort_index().to_string())
    print("\nDataset: trashnet")
    print(f"Source:  {source}")
    print(f"Splits:  {output}")
    print("Split proportions are approximately 70% train / 15% validation / 15% test.")


if __name__ == "__main__":
    main()
