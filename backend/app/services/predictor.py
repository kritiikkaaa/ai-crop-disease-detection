"""Thread-safe all-class inference with calibrated unknown-image rejection."""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock

import cv2
import numpy as np

IMAGE_SIZE = (224, 224)


def human_label(label: str) -> str:
    return label.replace("___", " — ").replace("_", " ")


def split_label(label: str) -> tuple[str, str]:
    crop, disease = label.split("___", 1)
    return crop.replace("_", " "), disease.replace("_", " ")


class Predictor:
    def __init__(self) -> None:
        self._model = None
        self._labels: list[str] = []
        self._calibration: dict = {}
        self._signature: tuple | None = None
        self._lock = Lock()

    @staticmethod
    def is_ready(model_path: str, labels_path: str, calibration_path: str) -> bool:
        return all(Path(path).is_file() for path in (model_path, labels_path, calibration_path))

    def _load(self, model_path: str, labels_path: str, calibration_path: str) -> None:
        model_file, labels_file, calibration_file = map(Path, (model_path, labels_path, calibration_path))
        if not self.is_ready(model_path, labels_path, calibration_path):
            raise FileNotFoundError("Model, label, or rejection-calibration artifact is missing.")
        signature = tuple((str(path), path.stat().st_mtime_ns) for path in (model_file, labels_file, calibration_file))
        with self._lock:
            if self._signature == signature:
                return
            import tensorflow as tf

            model = tf.keras.models.load_model(model_file, compile=False)
            labels = json.loads(labels_file.read_text(encoding="utf-8"))
            calibration = json.loads(calibration_file.read_text(encoding="utf-8"))
            if not isinstance(labels, list) or model.output_shape[-1] != len(labels):
                raise ValueError("The model output size does not match the class-name artifact.")
            if not calibration.get("per_predicted_class_confidence_thresholds"):
                raise ValueError("The rejection-calibration artifact is invalid.")
            self._model, self._labels, self._calibration, self._signature = model, labels, calibration, signature

    @staticmethod
    def _preprocess(raw: bytes) -> np.ndarray:
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("The uploaded file is not a readable image.")
        height, width = image.shape[:2]
        if height < 16 or width < 16:
            raise ValueError("The uploaded image is too small to analyze reliably.")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        return np.expand_dims(image.astype(np.float32) / 255.0, axis=0)

    def predict(self, raw: bytes, model_path: str, labels_path: str, calibration_path: str) -> dict:
        self._load(model_path, labels_path, calibration_path)
        probabilities = self._model.predict(self._preprocess(raw), verbose=0)[0]
        ranked = np.argsort(probabilities)[::-1][:3]
        index = int(ranked[0])
        label = self._labels[index]
        confidence = float(probabilities[index])
        safe = np.clip(probabilities, 1e-12, 1.0)
        entropy = float(-np.sum(safe * np.log(safe)) / np.log(len(probabilities)))
        thresholds = self._calibration["per_predicted_class_confidence_thresholds"]
        required_confidence = float(thresholds.get(label, self._calibration["global_confidence_threshold"]))
        classified = confidence >= required_confidence and entropy <= float(self._calibration["maximum_normalized_entropy"])
        result = {
            "status": "classified" if classified else "unknown",
            "label": label if classified else None,
            "prediction": human_label(label) if classified else None,
            "confidence": round(confidence * 100, 2),
            "normalized_entropy": round(entropy, 4),
            "message": "Image classified using the trained PlantVillage model." if classified else "Unknown / low confidence: this image cannot be classified reliably by this model.",
        }
        if classified:
            crop, disease = split_label(label)
            result.update({
                "crop": crop,
                "disease": disease,
                "top_predictions": [{"disease": human_label(self._labels[int(item)]), "confidence": round(float(probabilities[item]) * 100, 2)} for item in ranked],
            })
        return result


predictor = Predictor()
