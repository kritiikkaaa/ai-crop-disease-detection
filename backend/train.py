"""Train the all-class PlantVillage CNN from a verified split manifest.

Run dataset_pipeline.py first. This command does not alter PlantVillage images; it
loads paths from the generated manifest and writes only model/artifact outputs.
"""

from __future__ import annotations

import argparse
import json
import platform
from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from dataset_pipeline import DEFAULT_SEED, IMAGE_SIZE, load_manifest, make_tf_dataset


def build_model(class_count: int) -> tf.keras.Model:
    """Build a CNN whose Softmax size is determined by the dataset artifacts."""
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name="leaf_image")
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.10),
            tf.keras.layers.RandomZoom(0.10),
            tf.keras.layers.RandomTranslation(0.08, 0.08),
            tf.keras.layers.RandomBrightness(0.10, value_range=(0.0, 1.0)),
        ],
        name="training_augmentation",
    )
    x = augmentation(inputs)
    for filters, dropout in ((32, 0.10), (64, 0.15), (128, 0.20), (256, 0.25)):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.40)(x)
    outputs = tf.keras.layers.Dense(class_count, activation="softmax", name="class_probabilities")(x)
    model = tf.keras.Model(inputs, outputs, name="plantvillage_all_class_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


def class_weights(labels: np.ndarray, class_count: int) -> dict[int, float]:
    present = np.unique(labels)
    weights = compute_class_weight(class_weight="balanced", classes=present, y=labels)
    result = {int(label): float(weight) for label, weight in zip(present, weights)}
    if len(result) != class_count:
        raise ValueError("Every discovered class must be present in the training split.")
    return result


def training_metadata(
    classes: list[str], seed: int, batch_size: int, epochs: int, weights: dict[int, float]
) -> dict:
    return {
        "model_name": "plantvillage_all_class_cnn",
        "class_count": len(classes),
        "class_names": classes,
        "image_size": list(IMAGE_SIZE),
        "color_mode": "RGB",
        "normalization": "pixel / 255.0",
        "loss": "sparse_categorical_crossentropy",
        "optimizer": "Adam",
        "initial_learning_rate": 0.001,
        "batch_size": batch_size,
        "maximum_epochs": epochs,
        "seed": seed,
        "class_weights": weights,
        "augmentation_applies_only_during_fit": True,
        "callbacks": ["EarlyStopping", "ReduceLROnPlateau", "ModelCheckpoint"],
        "software": {"tensorflow": tf.__version__, "python": platform.python_version()},
        "physical_devices": [device.name for device in tf.config.list_physical_devices()],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the all-class PlantVillage CNN.")
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--models", type=Path, default=Path("model"))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    if not 1 <= args.epochs <= 25:
        raise SystemExit("--epochs must be between 1 and 25.")
    manifest = args.artifacts / "split_manifest.csv"
    classes_path = args.artifacts / "class_names.json"
    if not manifest.is_file() or not classes_path.is_file():
        raise SystemExit("Missing preparation artifacts. Run dataset_pipeline.py first.")

    classes = json.loads(classes_path.read_text(encoding="utf-8"))
    train_paths, train_labels = load_manifest(manifest, "train")
    validation_paths, validation_labels = load_manifest(manifest, "validation")
    _, test_labels = load_manifest(manifest, "test")
    if len(classes) != len(np.unique(train_labels)):
        raise SystemExit("The training manifest does not contain all discovered classes.")
    split_counts = Counter(np.concatenate((train_labels, validation_labels, test_labels)).tolist())
    print(f"Training on all detected classes: {len(classes)}")
    print("Complete class list and split image counts:")
    for index, label in enumerate(classes):
        print(f"  {label}: {split_counts[index]} total images")
    tf.keras.utils.set_random_seed(args.seed)
    train_ds = make_tf_dataset(train_paths, train_labels, args.batch_size, training=True, seed=args.seed)
    validation_ds = make_tf_dataset(validation_paths, validation_labels, args.batch_size, training=False, seed=args.seed)
    weights = class_weights(train_labels, len(classes))

    args.models.mkdir(parents=True, exist_ok=True)
    model_path = args.models / "crop_disease_model.keras"
    (args.artifacts / "model_metadata.json").write_text(
        json.dumps(training_metadata(classes, args.seed, args.batch_size, args.epochs, weights), indent=2),
        encoding="utf-8",
    )
    model = build_model(len(classes))
    model.summary()
    history = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=args.epochs,
        class_weight=weights,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.3, min_lr=1e-6),
            tf.keras.callbacks.ModelCheckpoint(model_path, monitor="val_accuracy", mode="max", save_best_only=True),
            tf.keras.callbacks.CSVLogger(args.artifacts / "training_log.csv"),
        ],
    )
    (args.artifacts / "training_history.json").write_text(json.dumps(history.history, indent=2), encoding="utf-8")
    (args.models / "class_names.json").write_text(json.dumps(classes, indent=2), encoding="utf-8")
    print(f"Saved best model to {model_path.resolve()}")


if __name__ == "__main__":
    main()
