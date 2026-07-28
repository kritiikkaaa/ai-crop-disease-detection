"""Train and evaluate a CNN on a PlantVillage directory dataset.

Expected dataset layout: DATASET_DIR/class_name/image.jpg.
Class labels are discovered from directory names and saved beside the model.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, precision_recall_fscore_support

IMAGE_SIZE = (224, 224)


def build_model(class_count: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    augmentation = tf.keras.Sequential([tf.keras.layers.RandomFlip("horizontal"), tf.keras.layers.RandomRotation(0.1), tf.keras.layers.RandomZoom(0.1)])
    x = augmentation(inputs)
    for filters in (32, 64, 128, 256):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Dropout(0.15)(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(class_count, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name="plant_disease_cnn")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def save_history(history: tf.keras.callbacks.History, output: Path) -> None:
    for metric, title in (("accuracy", "Model accuracy"), ("loss", "Model loss")):
        plt.figure(figsize=(8, 5)); plt.plot(history.history[metric], label="train"); plt.plot(history.history[f"val_{metric}"], label="validation")
        plt.title(title); plt.xlabel("Epoch"); plt.legend(); plt.tight_layout(); plt.savefig(output / f"{metric}.png", dpi=160); plt.close()


def evaluate(model: tf.keras.Model, dataset: tf.data.Dataset, labels: list[str], output: Path) -> None:
    y_true = np.concatenate([np.argmax(y.numpy(), axis=1) for _, y in dataset])
    y_pred = np.argmax(model.predict(dataset, verbose=0), axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    report = classification_report(y_true, y_pred, labels=range(len(labels)), target_names=labels, zero_division=0, output_dict=True)
    (output / "metrics.json").write_text(json.dumps({"precision": precision, "recall": recall, "f1_score": f1, "classification_report": report}, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(max(12, len(labels) * .35), max(10, len(labels) * .35)))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=labels, xticks_rotation=90, ax=ax, colorbar=False)
    fig.tight_layout(); fig.savefig(output / "confusion_matrix.png", dpi=180); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("dataset", type=Path); parser.add_argument("--epochs", type=int, default=30); parser.add_argument("--batch-size", type=int, default=32); parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(); tf.keras.utils.set_random_seed(args.seed)
    if not args.dataset.is_dir(): raise SystemExit(f"Dataset directory not found: {args.dataset}")
    train_ds = tf.keras.utils.image_dataset_from_directory(args.dataset, validation_split=.2, subset="training", seed=args.seed, image_size=IMAGE_SIZE, batch_size=args.batch_size, label_mode="categorical")
    val_ds = tf.keras.utils.image_dataset_from_directory(args.dataset, validation_split=.2, subset="validation", seed=args.seed, image_size=IMAGE_SIZE, batch_size=args.batch_size, label_mode="categorical", shuffle=False)
    labels = train_ds.class_names; root = Path(__file__).resolve().parent; model_dir, eval_dir = root / "models", root / "evaluation"; model_dir.mkdir(exist_ok=True); eval_dir.mkdir(exist_ok=True)
    (model_dir / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    autotune = tf.data.AUTOTUNE; train_ds = train_ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255., y), num_parallel_calls=autotune).prefetch(autotune); val_ds = val_ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255., y), num_parallel_calls=autotune).prefetch(autotune)
    model = build_model(len(labels)); checkpoint = model_dir / "model.keras"
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True), tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=3, factor=.3), tf.keras.callbacks.ModelCheckpoint(checkpoint, monitor="val_accuracy", save_best_only=True)])
    save_history(history, eval_dir); model = tf.keras.models.load_model(checkpoint); evaluate(model, val_ds, labels, eval_dir)

if __name__ == "__main__": main()
