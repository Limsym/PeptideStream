# tox_predictor.py

import joblib
import numpy as np

class ToxPredictor:
    def __init__(self, model_path="./models/toxicity_predictor.pkl"):
        self.model = joblib.load(model_path)

    def predict(self, embedding: np.ndarray) -> float:
        """返回毒性概率"""
        proba = self.model.predict_proba(embedding.reshape(1, -1))[:, 1]
        return proba[0]
