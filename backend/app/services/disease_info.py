from __future__ import annotations
import json
from pathlib import Path


class DiseaseInfoService:
    def __init__(self, data_path: str) -> None:
        self.data_path = Path(data_path)

    def get(self, label: str) -> dict:
        entries = json.loads(self.data_path.read_text(encoding="utf-8"))
        if label in entries:
            return entries[label]
        crop, condition = (label.split("___", 1) + ["Unknown condition"])[:2]
        healthy = "healthy" in condition.lower()
        return {
            "description": f"{crop.replace('_', ' ')} — {condition.replace('_', ' ')}.",
            "scientific_name": crop.replace("_", " "),
            "symptoms": (
                []
                if healthy
                else [
                    "Leaf discoloration, spots, lesions, or abnormal growth may be visible."
                ]
            ),
            "causes": (
                []
                if healthy
                else ["This label should be verified with local agricultural guidance."]
            ),
            "treatment": (
                ["No treatment is required; continue routine crop monitoring."]
                if healthy
                else [
                    "Consult a local extension service for a treatment approved for your crop and region."
                ]
            ),
            "prevention": [
                "Use clean tools, monitor plants regularly, and follow locally recommended crop-management practices."
            ],
        }
