# models/stability_predictor.py
import joblib
import numpy as np

class StabilityPredictor:
    def __init__(self, model_path="models/stability_predictor.pkl"):
        self.model = joblib.load(model_path)

    def predict(self, embedding: np.ndarray) -> float:
        """
        输入单个蛋白 embedding，返回预测的 ΔΔG (稳定性变化)
        正值：不稳定；负值：稳定化
        """
        return float(self.model.predict(embedding.reshape(1, -1))[0])
