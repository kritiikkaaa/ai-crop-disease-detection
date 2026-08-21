"""Thread-safe TensorFlow inference and OpenCV image preprocessing."""

from __future__ import annotations
import json
from pathlib import Path
from threading import Lock
import cv2
import numpy as np


class Predictor:
    def __init__(self) -> None:
        self._model = None
        self._labels: list[str] | None = None
        self._signature: tuple[str, float, str, float] | None = None
        self._lock = Lock()

    def is_ready(self, model_path: str, labels_path: str) -> bool:
        return Path(model_path).is_file() and Path(labels_path).is_file()

    def _load(self, model_path: str, labels_path: str) -> None:
        model_file, labels_file = Path(model_path), Path(labels_path)
        if not self.is_ready(model_path, labels_path):
            raise FileNotFoundError(model_path)
        signature = (
            str(model_file),
            model_file.stat().st_mtime,
            str(labels_file),
            labels_file.stat().st_mtime,
        )
        with self._lock:
            if self._signature == signature:
                return
            import tensorflow as tf

            self._model = tf.keras.models.load_model(model_file, compile=False)
            loaded = json.loads(labels_file.read_text(encoding="utf-8"))
            self._labels = loaded if isinstance(loaded, list) else loaded["labels"]
            self._signature = signature

    @staticmethod
    def _preprocess(raw: bytes) -> np.ndarray:
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("The uploaded file is not a readable image.")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        # Match the paper-aligned training preprocessing: RGB pixels in [0, 1].
        return np.expand_dims(image.astype(np.float32) / 255.0, axis=0)

    def predict(self, raw: bytes, model_path: str, labels_path: str) -> dict:
        self._load(model_path, labels_path)
        probabilities = self._model.predict(self._preprocess(raw), verbose=0)[0]
        ranked = np.argsort(probabilities)[::-1][:3]
        label = self._labels[int(ranked[0])]
        return {
            "label": label,
            "disease": label.replace("___", " — ").replace("_", " "),
            "confidence": round(float(probabilities[ranked[0]]) * 100, 2),
            "top_predictions": [
                {
                    "disease": self._labels[int(i)]
                    .replace("___", " — ")
                    .replace("_", " "),
                    "confidence": round(float(probabilities[i]) * 100, 2),
                }
                for i in ranked
            ],
        }


predictor = Predictor()
