"""Exact-pixel registry for the curated Custom dataset."""
from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
CUSTOM_TRAIN_DIR = ROOT / "data" / "custom_images" / "train"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def canonical_image_hash(image: Image.Image) -> str:
    """Hash orientation-corrected RGB pixels and dimensions, not filename/metadata."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    digest = hashlib.sha256()
    digest.update(f"{image.width}x{image.height}:RGB\0".encode("utf-8"))
    digest.update(image.tobytes())
    return digest.hexdigest()


def image_file_hash(path: Path) -> str:
    with Image.open(path) as image:
        return canonical_image_hash(image)


def build_custom_registry(root: Path = CUSTOM_TRAIN_DIR) -> dict[str, str]:
    """Map exact decoded image content to its folder label.

    If identical pixels appear under conflicting labels, map that hash to
    'trash' rather than silently selecting one of the conflicting labels.
    """
    registry: dict[str, str] = {}
    conflicts: set[str] = set()
    if not root.is_dir():
        return registry

    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir() or class_dir.name.startswith(".") or class_dir.name == "__MACOSX":
            continue
        label = class_dir.name.strip().lower().replace("_", " ").replace("-", " ")
        for path in sorted(class_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            try:
                digest = image_file_hash(path)
            except (OSError, ValueError):
                continue
            if digest in registry and registry[digest] != label:
                conflicts.add(digest)
            else:
                registry[digest] = label

    for digest in conflicts:
        registry[digest] = "trash"
    return registry
