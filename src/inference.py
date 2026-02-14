
from __future__ import annotations


from dataclasses import dataclass
from typing import Any, Dict, List, Mapping

from src.data_loader import (load_artifact, load_model)

from src.feature_engineering import payload_to_frame


FEATURE_NAMES: List[str] = [
    "social_energy",
    "alone_time_preference",
    "talkativeness",
    "deep_reflection",
    "group_comfort",
    "party_liking",
    "leadership",
    "risk_taking",
    "public_speaking_comfort",
    "excitement_seeking",
    "adventurousness",
    "reading_habit",
]


@dataclass(frozen=True)
class Predictor:
    model: Any
    label_encoder: Any

    @staticmethod
    def load() -> "Predictor":
        model = load_model("log_reg_12Features.pkl", model_dir="models")
        label_encoder = load_artifact("label_encoder.joblib", split_dir="data/artifacts")
        return Predictor(model=model,  label_encoder=label_encoder)

    def predict_proba(self, payload: Mapping[str, Any]) -> Dict[str, float]:
        """
        payload: словарь из 12 фич (0..10), строго по FEATURE_NAMES.
        Возвращает вероятности по человекочитаемым классам.
        """
        X = payload_to_frame(payload, FEATURE_NAMES) 

        proba = self.model.predict_proba(X)[0]


       
        enc_classes = list(getattr(self.model, "classes_", range(len(proba))))
        labels = self.label_encoder.inverse_transform([int(c) for c in enc_classes])

        return {str(labels[i]): float(proba[i]) for i in range(len(labels))}

