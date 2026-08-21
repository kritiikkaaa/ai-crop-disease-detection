"""Train and evaluate the CropVision CNN on the verified PlantVillage classes.

The source dataset is never modified. A deterministic 70/15/15 stratified split is
created from ``raw/color`` and only the training pipeline includes augmentation.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split

IMAGE_SIZE = (224, 224)
SEED = 42
SELECTED_CLASSES = [
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___healthy",
]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def collect_samples(dataset_dir: Path) -> tuple[list[Path], list[str]]:
    """Validate the confirmed folders and collect only real image files."""
    paths, labels = [], []
    for label in SELECTED_CLASSES:
        class_dir = dataset_dir / label
        if not class_dir.is_dir():
            raise SystemExit(f"Confirmed class directory is missing: {class_dir}")
        images = sorted(p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if not images:
            raise SystemExit(f"No supported images found in: {class_dir}")
        paths.extend(images)
        labels.extend([label] * len(images))
    return paths, labels


def split_samples(paths: list[Path], labels: list[str], seed: int):
    """Create reproducible stratified 70% train, 15% validation, 15% test splits."""
    train_paths, remaining_paths, train_labels, remaining_labels = train_test_split(
        paths, labels, test_size=0.30, random_state=seed, stratify=labels
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        remaining_paths, remaining_labels, test_size=0.50,
        random_state=seed, stratify=remaining_labels,
    )
    return {
        "train": (train_paths, train_labels),
        "validation": (val_paths, val_labels),
        "test": (test_paths, test_labels),
    }


def save_distribution(splits: dict, output: Path) -> None:
    rows = []
    for split, (_, labels) in splits.items():
        counts = Counter(labels)
        for label in SELECTED_CLASSES:
            rows.append({"split": split, "class": label, "images": counts[label]})
    with (output / "dataset_distribution.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["split", "class", "images"])
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        split: {"total": len(labels), "classes": dict(Counter(labels))}
        for split, (_, labels) in splits.items()
    }
    (output / "dataset_distribution.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def make_dataset(paths: list[Path], labels: list[str], training: bool, batch_size: int):
    label_ids = np.array([SELECTED_CLASSES.index(label) for label in labels], dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices(([str(path) for path in paths], label_ids))
    if training:
        dataset = dataset.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)

    def load_image(path: tf.Tensor, label: tf.Tensor):
        image = tf.io.read_file(path)
        image = tf.io.decode_image(image, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, IMAGE_SIZE, antialias=True)
        image = tf.cast(image, tf.float32) / 255.0
        return image, tf.one_hot(label, depth=len(SELECTED_CLASSES))

    return dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_model(class_count: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name="leaf_image")
    # Keras applies this only while fitting; validation, test, and inference are unchanged.
    x = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.10),
        tf.keras.layers.RandomZoom(0.10),
        tf.keras.layers.RandomTranslation(0.08, 0.08),
    ], name="training_augmentation")(inputs)
    for filters in (32, 64, 128, 256):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Dropout(0.15)(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.40)(x)
    outputs = tf.keras.layers.Dense(class_count, activation="softmax", name="disease")(x)
    model = tf.keras.Model(inputs, outputs, name="crop_disease_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_history(history: tf.keras.callbacks.History, output: Path) -> None:
    (output / "training_history.json").write_text(
        json.dumps(history.history, indent=2), encoding="utf-8"
    )
    for metric, filename, title in (
        ("accuracy", "training_accuracy.png", "Training and validation accuracy"),
        ("loss", "training_loss.png", "Training and validation loss"),
    ):
        plt.figure(figsize=(8, 5))
        plt.plot(history.history[metric], label="Training")
        plt.plot(history.history[f"val_{metric}"], label="Validation")
        plt.title(title)
        plt.xlabel("Epoch")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output / filename, dpi=160)
        plt.close()


def evaluate(model: tf.keras.Model, test_ds, output: Path) -> None:
    y_true = np.concatenate([np.argmax(labels.numpy(), axis=1) for _, labels in test_ds])
    y_pred = np.argmax(model.predict(test_ds, verbose=0), axis=1)
    accuracy = float(np.mean(y_true == y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    report_text = classification_report(y_true, y_pred, target_names=SELECTED_CLASSES, zero_division=0)
    (output / "classification_report.txt").write_text(report_text, encoding="utf-8")
    (output / "metrics.json").write_text(json.dumps({
        "test_accuracy": accuracy, "weighted_precision": float(precision),
        "weighted_recall": float(recall), "weighted_f1_score": float(f1),
    }, indent=2), encoding="utf-8")
    fig, axis = plt.subplots(figsize=(12, 10))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=SELECTED_CLASSES,
        xticks_rotation=35, colorbar=False, ax=axis,
    )
    fig.tight_layout()
    fig.savefig(output / "confusion_matrix.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path, help="Path to PlantVillage raw/color")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if not args.dataset.is_dir():
        raise SystemExit(f"Dataset directory not found: {args.dataset}")
    if args.epochs < 1 or args.epochs > 25:
        raise SystemExit("--epochs must be between 1 and 25, as specified in the paper.")

    tf.keras.utils.set_random_seed(args.seed)
    paths, labels = collect_samples(args.dataset)
    root = Path(__file__).resolve().parent
    model_dir, results_dir = root / "models", root / "evaluation"
    model_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    splits = split_samples(paths, labels, args.seed)
    save_distribution(splits, results_dir)
    train_ds = make_dataset(*splits["train"], training=True, batch_size=args.batch_size)
    validation_ds = make_dataset(*splits["validation"], training=False, batch_size=args.batch_size)
    test_ds = make_dataset(*splits["test"], training=False, batch_size=args.batch_size)

    model_path = model_dir / "crop_disease_model.keras"
    model = build_model(len(SELECTED_CLASSES))
    history = model.fit(train_ds, validation_data=validation_ds, epochs=args.epochs, callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.3),
        tf.keras.callbacks.ModelCheckpoint(model_path, monitor="val_accuracy", mode="max", save_best_only=True),
    ])
    (model_dir / "labels.json").write_text(json.dumps(SELECTED_CLASSES, indent=2), encoding="utf-8")
    save_history(history, results_dir)
    evaluate(tf.keras.models.load_model(model_path), test_ds, results_dir)


if __name__ == "__main__":
    main()
