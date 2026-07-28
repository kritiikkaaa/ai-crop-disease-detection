from flask import Blueprint, current_app, jsonify, request
from werkzeug.exceptions import BadRequest, ServiceUnavailable
from app.services.disease_info import DiseaseInfoService
from app.services.predictor import predictor

predict_bp = Blueprint("predict", __name__)


@predict_bp.post("/predict")
def predict():
    image = request.files.get("image")
    if image is None or not image.filename:
        raise BadRequest("Provide an image file in the 'image' field.")
    if image.mimetype not in {"image/jpeg", "image/png", "image/webp"}:
        raise BadRequest("Only JPEG, PNG, and WEBP images are accepted.")
    try:
        result = predictor.predict(image.read(), current_app.config["MODEL_PATH"], current_app.config["LABELS_PATH"])
    except FileNotFoundError:
        raise ServiceUnavailable("Model artifacts are unavailable. Train the model before making predictions.")
    info = DiseaseInfoService(current_app.config["DISEASE_INFO_PATH"]).get(result["label"])
    return jsonify({"disease": result["disease"], "confidence": result["confidence"], "top_predictions": result["top_predictions"], **info})
