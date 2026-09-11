"""Calibrate confidence and entropy rejection limits from validation predictions.

The resulting thresholds are empirical properties of the trained model and its
untouched validation split. They are not fixed application constants.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from dataset_pipeline import load_manifest, make_tf_dataset


def normalized_entropy(probabilities: np.ndarray) -> np.ndarray:
    safe = np.clip(probabilities, 1e-12, 1.0)
    return -np.sum(safe * np.log(safe), axis=1) / np.log(probabilities.shape[1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate unknown-image rejection from validation data.")
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--models", type=Path, default=Path("model"))
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    manifest = args.artifacts / "split_manifest.csv"
    classes_path = args.models / "class_names.json"
    model_path = args.models / "crop_disease_model.keras"
    for required in (manifest, classes_path, model_path):
        if not required.is_file():
            raise SystemExit(f"Required calibration input is missing: {required}")

    classes = json.loads(classes_path.read_text(encoding="utf-8"))
    paths, y_true = load_manifest(manifest, "validation")
    dataset = make_tf_dataset(paths, y_true, args.batch_size, training=False, seed=42)
    model = tf.keras.models.load_model(model_path, compile=False)
    probabilities = model.predict(dataset, verbose=1)
    predicted = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1)
    entropy = normalized_entropy(probabilities)
    correct = predicted == y_true
    if not np.any(correct):
        raise SystemExit("No correct validation predictions; rejection cannot be calibrated.")

    # Retain roughly 95% of correctly classified validation images on each signal.
    global_confidence = float(np.quantile(confidence[correct], 0.05))
    maximum_entropy = float(np.quantile(entropy[correct], 0.95))
    per_class_confidence = {}
    for index, label in enumerate(classes):
        values = confidence[correct & (predicted == index)]
        if len(values):
            per_class_confidence[label] = float(np.quantile(values, 0.05))

    calibration = {
        "method": "validation-calibrated confidence and normalized entropy rejection",
        "validation_image_count": int(len(y_true)),
        "correct_validation_prediction_count": int(np.sum(correct)),
        "correct_validation_accuracy": float(np.mean(correct)),
        "global_confidence_threshold": global_confidence,
        "per_predicted_class_confidence_thresholds": per_class_confidence,
        "maximum_normalized_entropy": maximum_entropy,
        "confidence_quantile": 0.05,
        "entropy_quantile": 0.95,
        "limitations": (
            "This is a confidence-based rejection heuristic calibrated on PlantVillage validation images. "
            "It can reduce unreliable classifications but cannot guarantee detection of every out-of-distribution image."
        ),
    }
    output = args.artifacts / "rejection_calibration.json"
    output.write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    print(json.dumps(calibration, indent=2))


if __name__ == "__main__":
    main()
