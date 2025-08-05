# multi_feature_evaluator.py
from models.tox_predictor import ToxPredictor
# from models.stability_predictor import StabilityPredictor
# from models.efficacy_predictor import EfficacyPredictor
from embedding.esm_embedding_extractor import extract_protein_embedding, load_esm_components

class MultiFeatureEvaluator:
    def __init__(self, esm_path: str):
        self.tox_model = ToxPredictor("models/toxicity_predictor.pkl")
        # self.stab_model = StabilityPredictor("models/stability_predictor.pkl")
        # self.eff_model = EfficacyPredictor("models/efficacy_predictor.pkl")
        self.model, self.tokenizer = load_esm_components(esm_path)

    def evaluate_sequence(self, seq: str) -> dict:
        emb = extract_protein_embedding(seq, self.model, self.tokenizer)
        results = {
            "toxicity": self.tox_model.predict(emb),
            # "stability": self.stab_model.predict(emb),
            # "efficacy": self.eff_model.predict(emb),
        }
        return results

    def is_candidate(self, scores: dict) -> bool:
        return scores["toxicity"] < 0.3  # 可以扩展成可配置的规则
