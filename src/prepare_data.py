"""Create reproducible train/validation/test manifests for TrashNet or RealWaste.

TrashNet: data/raw/dataset-resized/<class>/*.jpg
RealWaste: extract the UCI archive into data/raw/realwaste/; this script
finds the nested RealWaste class directory automatically.

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


def locate_source(dataset: str) -> Path:
    if dataset == "trashnet":
        source = RAW_DIR / "dataset-resized"
        if source.is_dir() and len(class_directories(source)) >= 2:
            return source
        raise SystemExit(
            f"TrashNet folders not found at {source}. "
            "Extract the resized archive so each class has its own folder."
        )

    # Search only within the RealWaste extraction directory so that we cannot
    # accidentally select TrashNet when RealWaste has not been downloaded.
    search_root = RAW_DIR / "realwaste"
    if not search_root.exists():
        raise SystemExit(
            f"RealWaste extraction directory not found: {search_root}\n"
            "Download the UCI RealWaste archive and extract it into data/raw/realwaste/."
        )

    candidates = [search_root]
    candidates.extend(p for p in search_root.rglob("*") if p.is_dir())
    valid: list[tuple[int, int, Path]] = []
    for candidate in candidates:
        class_dirs = class_directories(candidate)
        if len(class_dirs) >= 2:
            total = sum(len(image_files(d)) for d in class_dirs)
            # Prefer the deepest class-root with the greatest image count.
            valid.append((total, len(class_dirs), candidate))
    if not valid:
        raise SystemExit(
            f"Could not find class folders under {search_root}. "
            "Expected a RealWaste folder containing folders such as Cardboard, Glass and Metal."
        )
    valid.sort(key=lambda item: (item[0], item[1], len(item[2].parts)), reverse=True)
    return valid[0][2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=["trashnet", "realwaste"], default="trashnet",
        help="Which dataset to prepare (default: trashnet).",
    )
    args = parser.parse_args()

    source = locate_source(args.dataset)
    # Preserve the paths/manifests used by the already-trained TrashNet models.
    output = OUTPUT_ROOT if args.dataset == "trashnet" else OUTPUT_ROOT / "realwaste"

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
        "dataset": args.dataset,
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
    print(f"\nDataset: {args.dataset}")
    print(f"Source:  {source}")
    print(f"Splits:  {output}")
    print("Split proportions are approximately 70% train / 15% validation / 15% test.")


if __name__ == "__main__":
    main()
