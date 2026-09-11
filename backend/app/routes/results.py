"""Expose generated evaluation artifacts without embedding their values in code."""

from __future__ import annotations

import json
from pathlib import Path

from flask import Blueprint, current_app, jsonify, send_from_directory
from werkzeug.exceptions import NotFound

results_bp = Blueprint("results", __name__)


@results_bp.get("/results")
def results():
    root = Path(current_app.config["RESULTS_PATH"])
    metrics_path = root / "metrics.json"
    report_path = root / "classification_report.json"
    metadata_path = Path(current_app.config["MODEL_METADATA_PATH"])
    if not metrics_path.is_file() or not report_path.is_file():
        raise NotFound("Evaluation results are unavailable. Evaluate a trained model first.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else None
    class_names = set(metadata.get("class_names", [])) if metadata else set()
    class_metrics = [
        {"class": label, **values}
        for label, values in report.items()
        if label in class_names and isinstance(values, dict) and "precision" in values and "support" in values
    ]
    graphs_dir = root / "graphs"
    graphs = [path.name for path in sorted(graphs_dir.glob("*.png"))] if graphs_dir.is_dir() else []
    return jsonify({
        "metrics": json.loads(metrics_path.read_text(encoding="utf-8")),
        "class_metrics": class_metrics,
        "model_metadata": metadata,
        "graphs": graphs,
    })


@results_bp.get("/results/graphs/<path:filename>")
def graph(filename: str):
    root = Path(current_app.config["RESULTS_PATH"]) / "graphs"
    requested = (root / filename).resolve()
    if root.resolve() not in requested.parents or not requested.is_file() or requested.suffix != ".png":
        raise NotFound("Evaluation graph not found.")
    return send_from_directory(root, filename)
