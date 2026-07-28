from flask import Blueprint, current_app, jsonify
from app.services.predictor import predictor

health_bp = Blueprint("health", __name__)


@health_bp.route("/health", methods=["GET", "POST"])
def health():
    ready = predictor.is_ready(current_app.config["MODEL_PATH"], current_app.config["LABELS_PATH"])
    return jsonify({"status": "healthy" if ready else "model_unavailable", "model_ready": ready}), 200 if ready else 503
