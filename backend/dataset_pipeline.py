"""Reproducible PlantVillage data preparation for all discovered RGB classes.

This module only reads the original dataset. It writes manifests and metadata, never
copies, moves, or deletes source images.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split

IMAGE_SIZE = (224, 224)
VALID_SUFFIXES = {".jpg", ".jpeg", ".png"}
DEFAULT_SEED = 42


@dataclass(frozen=True)
class ImageRecord:
    path: str
    label: str
    sha256: str


def discover_classes(dataset_root: Path) -> list[str]:
    classes = sorted(path.name for path in dataset_root.iterdir() if path.is_dir())
    if len(classes) < 2:
        raise ValueError(f"Expected at least two class directories in {dataset_root}.")
    return classes


def scan_dataset(dataset_root: Path, classes: Iterable[str]) -> tuple[list[ImageRecord], dict]:
    """Validate files and retain one deterministic representative per exact hash."""
    records: list[ImageRecord] = []
    hashes: dict[str, list[ImageRecord]] = defaultdict(list)
    unreadable: list[dict[str, str]] = []
    invalid_files: list[str] = []
    formats, modes, dimensions = Counter(), Counter(), Counter()

    for label in classes:
        class_dir = dataset_root / label
        for path in sorted(class_dir.iterdir()):
            if not path.is_file():
                continue
            if path.suffix.lower() not in VALID_SUFFIXES:
                invalid_files.append(str(path))
                continue
            try:
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    formats[image.format or "UNKNOWN"] += 1
                    modes[image.mode] += 1
                    dimensions[f"{image.width}x{image.height}"] += 1
            except (UnidentifiedImageError, OSError, ValueError) as error:
                unreadable.append({"path": str(path), "error": str(error)})
                continue
            record = ImageRecord(
                path=str(path.resolve()), label=label,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            hashes[record.sha256].append(record)

    duplicate_groups = [group for group in hashes.values() if len(group) > 1]
    duplicate_paths = {record.path for group in duplicate_groups for record in group[1:]}
    for group in hashes.values():
        records.append(group[0])  # input is sorted; this is deterministic.
    records.sort(key=lambda record: (record.label, record.path))
    summary = {
        "raw_valid_image_count": len(records) + len(duplicate_paths),
        "usable_unique_image_count": len(records),
        "unreadable_image_count": len(unreadable),
        "invalid_file_count": len(invalid_files),
        "duplicate_group_count": len(duplicate_groups),
        "duplicate_file_count_excluded_from_manifest": len(duplicate_paths),
        "formats": dict(sorted(formats.items())),
        "color_modes": dict(sorted(modes.items())),
        "dimensions": dict(sorted(dimensions.items())),
        "unreadable_images": unreadable,
        "invalid_files": invalid_files,
        "duplicate_groups": [[asdict(record) for record in group] for group in duplicate_groups],
    }
    return records, summary


def stratified_splits(records: list[ImageRecord], seed: int) -> dict[str, list[ImageRecord]]:
    labels = [record.label for record in records]
    train, holdout = train_test_split(
        records, test_size=0.30, random_state=seed, stratify=labels
    )
    validation, test = train_test_split(
        holdout, test_size=0.50, random_state=seed,
        stratify=[record.label for record in holdout],
    )
    return {"train": train, "validation": validation, "test": test}


def write_artifacts(
    output_dir: Path,
    dataset_root: Path,
    classes: list[str],
    splits: dict[str, list[ImageRecord]],
    scan_summary: dict,
    seed: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    class_indices = {label: index for index, label in enumerate(classes)}
    (output_dir / "class_names.json").write_text(json.dumps(classes, indent=2), encoding="utf-8")
    (output_dir / "class_indices.json").write_text(json.dumps(class_indices, indent=2), encoding="utf-8")
    preprocessing = {
        "image_size": list(IMAGE_SIZE), "color_mode": "RGB", "normalization": "pixel / 255.0",
        "training_augmentation": {
            "horizontal_flip": True, "rotation_factor": 0.10, "zoom_factor": 0.10,
            "translation_height_factor": 0.08, "translation_width_factor": 0.08,
            "brightness_factor": 0.10,
        },
        "validation_test_inference_augmentation": False,
    }
    (output_dir / "preprocessing_config.json").write_text(json.dumps(preprocessing, indent=2), encoding="utf-8")

    rows = []
    distribution = {}
    for split, records in splits.items():
        counts = Counter(record.label for record in records)
        distribution[split] = {"total": len(records), "per_class": dict(counts)}
        rows.extend({
            "split": split, "path": record.path, "label": record.label,
            "class_index": class_indices[record.label], "sha256": record.sha256,
        } for record in records)
    with (output_dir / "split_manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["split", "path", "label", "class_index", "sha256"])
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "dataset_root": str(dataset_root.resolve()), "seed": seed, "split_ratio": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "class_count": len(classes), "class_names": classes,
        "crop_count": len({label.split("___", 1)[0] for label in classes}),
        "scan": scan_summary, "splits": distribution,
    }
    (output_dir / "dataset_summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_manifest(manifest_path: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    paths, labels = [], []
    with manifest_path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row["split"] == split:
                paths.append(row["path"])
                labels.append(int(row["class_index"]))
    if not paths:
        raise ValueError(f"No records found for split '{split}' in {manifest_path}.")
    return np.asarray(paths), np.asarray(labels, dtype=np.int32)


def make_tf_dataset(paths, labels, batch_size: int, training: bool, seed: int):
    """Create the common non-augmenting loader used by training and evaluation."""
    import tensorflow as tf

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)

    def decode(path, label):
        image = tf.io.read_file(path)
        image = tf.io.decode_image(image, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, IMAGE_SIZE, antialias=True)
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    return dataset.map(decode, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and split PlantVillage RGB classes.")
    parser.add_argument("dataset", type=Path, help="PlantVillage raw/color directory")
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    if not args.dataset.is_dir():
        raise SystemExit(f"Dataset directory not found: {args.dataset}")
    classes = discover_classes(args.dataset)
    records, scan_summary = scan_dataset(args.dataset, classes)
    splits = stratified_splits(records, args.seed)
    available_counts = Counter()
    for label in classes:
        available_counts[label] = sum(
            1 for path in (args.dataset / label).iterdir()
            if path.is_file() and path.suffix.lower() in VALID_SUFFIXES
        )
    print(f"Dataset root: {args.dataset.resolve()}")
    print(f"Detected classes: {len(classes)}")
    print("Complete class list and available image counts:")
    for label in classes:
        print(f"  {label}: {available_counts[label]}")
    print(f"Total valid images: {sum(available_counts.values())}")
    write_artifacts(args.output, args.dataset, classes, splits, scan_summary, args.seed)
    print(json.dumps({
        "classes": len(classes), "unique_images": len(records),
        "train": len(splits["train"]), "validation": len(splits["validation"]), "test": len(splits["test"]),
        "artifacts": str(args.output.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
