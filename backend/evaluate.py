"""Evaluate a trained PlantVillage model on the untouched manifest test split."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from dataset_pipeline import load_manifest, make_tf_dataset


def save_history_graphs(history_path: Path, results_dir: Path, graphs_dir: Path) -> None:
    if not history_path.is_file():
        return
    history = json.loads(history_path.read_text(encoding="utf-8"))
    for metric, filename, title in (
        ("accuracy", "training_accuracy.png", "Training vs validation accuracy"),
        ("loss", "training_loss.png", "Training vs validation loss"),
    ):
        if metric not in history or f"val_{metric}" not in history:
            continue
        plt.figure(figsize=(8, 5))
        plt.plot(history[metric], label="Training")
        plt.plot(history[f"val_{metric}"], label="Validation")
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(metric.replace("_", " ").title())
        plt.legend()
        plt.tight_layout()
        plt.savefig(results_dir / filename, dpi=180)
        plt.savefig(graphs_dir / filename, dpi=180)
        plt.close()


def save_distribution_graph(manifest: Path, classes: list[str], graphs_dir: Path) -> None:
    counts = Counter()
    with manifest.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            counts[row["label"]] += 1
    plt.figure(figsize=(14, max(8, len(classes) * 0.32)))
    plt.barh(classes, [counts[label] for label in classes])
    plt.title("Class distribution after duplicate exclusion")
    plt.xlabel("Images")
    plt.tight_layout()
    plt.savefig(graphs_dir / "class_distribution.png", dpi=180)
    plt.close()


def save_per_class_graph(report: dict, classes: list[str], metric: str, graphs_dir: Path) -> None:
    values = [report[label][metric] for label in classes]
    plt.figure(figsize=(14, max(8, len(classes) * 0.32)))
    plt.barh(classes, values)
    plt.xlim(0, 1)
    plt.title(f"Per-class {metric.replace('_', ' ').title()}")
    plt.xlabel(metric.replace("_", " ").title())
    plt.tight_layout()
    plt.savefig(graphs_dir / f"per_class_{metric}.png", dpi=180)
    plt.close()


def write_report_csv(report: dict, classes: list[str], path: Path) -> None:
    columns = ["class", "precision", "recall", "f1_score", "support"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for label in classes:
            entry = report[label]
            writer.writerow({
                "class": label, "precision": entry["precision"], "recall": entry["recall"],
                "f1_score": entry["f1-score"], "support": entry["support"],
            })


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the best all-class PlantVillage model.")
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--models", type=Path, default=Path("model"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    manifest = args.artifacts / "split_manifest.csv"
    class_path = args.models / "class_names.json"
    model_path = args.models / "crop_disease_model.keras"
    for required in (manifest, class_path, model_path):
        if not required.is_file():
            raise SystemExit(f"Required evaluation input is missing: {required}")
    classes = json.loads(class_path.read_text(encoding="utf-8"))
    test_paths, test_labels = load_manifest(manifest, "test")
    test_ds = make_tf_dataset(test_paths, test_labels, args.batch_size, training=False, seed=42)
    model = tf.keras.models.load_model(model_path, compile=False)
    if model.output_shape[-1] != len(classes):
        raise SystemExit("Model output size does not match the saved class-name artifact.")

    probabilities = model.predict(test_ds, verbose=1)
    predictions = np.argmax(probabilities, axis=1)
    labels = list(range(len(classes)))
    report = classification_report(
        test_labels, predictions, labels=labels, target_names=classes,
        output_dict=True, zero_division=0,
    )
    macro = precision_recall_fscore_support(test_labels, predictions, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(test_labels, predictions, average="weighted", zero_division=0)
    metrics = {
        "test_image_count": int(len(test_labels)),
        "accuracy": float(accuracy_score(test_labels, predictions)),
        "macro_precision": float(macro[0]), "macro_recall": float(macro[1]), "macro_f1": float(macro[2]),
        "weighted_precision": float(weighted[0]), "weighted_recall": float(weighted[1]), "weighted_f1": float(weighted[2]),
    }

    graphs_dir = args.results / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (args.results / "classification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    report_text = classification_report(
        test_labels, predictions, labels=labels, target_names=classes, zero_division=0,
    )
    (args.results / "classification_report.txt").write_text(report_text, encoding="utf-8")
    write_report_csv(report, classes, args.results / "classification_report.csv")

    matrix = confusion_matrix(test_labels, predictions, labels=labels)
    np.savetxt(args.results / "confusion_matrix.csv", matrix, delimiter=",", fmt="%d")
    normalized = confusion_matrix(test_labels, predictions, labels=labels, normalize="true")
    np.savetxt(args.results / "confusion_matrix_normalized.csv", normalized, delimiter=",", fmt="%.6f")
    for values, filename, title, fmt in (
        (matrix, "confusion_matrix.png", "Test-set confusion matrix", "d"),
        (normalized, "confusion_matrix_normalized.png", "Normalized test-set confusion matrix", ".2f"),
    ):
        fig, axis = plt.subplots(figsize=(22, 20))
        display = ConfusionMatrixDisplay(values, display_labels=classes)
        display.plot(ax=axis, xticks_rotation=90, include_values=False, colorbar=True, cmap="Blues")
        axis.set_title(title)
        fig.tight_layout()
        fig.savefig(args.results / filename, dpi=220)
        fig.savefig(graphs_dir / filename, dpi=220)
        plt.close(fig)
    for metric in ("precision", "recall", "f1-score"):
        filename_metric = metric.replace("-score", "") if metric == "f1-score" else metric
        save_per_class_graph(report, classes, metric, graphs_dir)
        source = graphs_dir / f"per_class_{metric}.png"
        if source.exists() and source.name != f"per_class_{filename_metric}.png":
            source.rename(graphs_dir / f"per_class_{filename_metric}.png")
    save_distribution_graph(manifest, classes, graphs_dir)
    save_history_graphs(args.artifacts / "training_history.json", args.results, graphs_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
