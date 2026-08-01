"""Application factory for the crop disease detection API."""
from __future__ import annotations

import logging
import os
from pathlib import Path

from flask import Flask, jsonify
from flask_cors import CORS
from werkzeug.exceptions import HTTPException


def create_app() -> Flask:
    app = Flask(__name__)
    root = Path(__file__).resolve().parents[1]
    app.config.from_mapping(
        MAX_CONTENT_LENGTH=int(os.getenv("MAX_CONTENT_LENGTH", 8 * 1024 * 1024)),
        MODEL_PATH=os.getenv("MODEL_PATH", str(root / "models" / "model.keras")),
        LABELS_PATH=os.getenv("LABELS_PATH", str(root / "models" / "labels.json")),
        DISEASE_INFO_PATH=str(root / "app" / "data" / "disease_info.json"),
    )
    origins = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:5174,http://127.0.0.1:5174",
    ).split(",")
    CORS(app, resources={r"/*": {"origins": origins}})
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    from app.routes.health import health_bp
    from app.routes.predict import predict_bp
    app.register_blueprint(health_bp)
    app.register_blueprint(predict_bp)

    @app.get("/")
    def index():
        return jsonify({"service": "Crop Disease Detection API", "status": "online"})

    @app.errorhandler(HTTPException)
    def handle_http_error(error: HTTPException):
        return jsonify({"error": error.description}), error.code

    @app.errorhandler(Exception)
    def handle_unexpected_error(error: Exception):
        app.logger.exception("Unhandled API error")
        return jsonify({"error": "An unexpected server error occurred."}), 500

    return app