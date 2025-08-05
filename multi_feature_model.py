# multi_feature_model.py

import numpy as np
from typing import Dict
from models.tox_model_trainer import predict_toxicity
# from stability_model import predict_stability
# from affinity_model import predict_affinity
# from efficacy_model import predict_efficacy

# 如果有 tokenizer / model 对象可以共用，可从这里导入
# from shared_model_loader import esm_model, esm_tokenizer

def predict_all_properties(sequence: str) -> Dict[str, float]:
    """
    输入一条氨基酸序列，输出多个性质的预测结果。
    目前支持：toxicity，后续拓展：stability、binding_affinity、bioactivity
    """

    # ✅ 毒性预测
    try:
        toxicity_score = predict_toxicity(sequence)
    except Exception as e:
        print(f"[Error] Toxicity prediction failed: {e}")
        toxicity_score = np.nan

    # 🧱 稳定性预测（待实现）
    try:
        stability_score = 0.5  # predict_stability(sequence)
    except:
        stability_score = np.nan

    # 🧱 亲和力预测（待实现）
    try:
        affinity_score = 0.5  # predict_affinity(sequence)
    except:
        affinity_score = np.nan

    # 🧱 药效预测（待实现）
    try:
        efficacy_score = 0.5  # predict_efficacy(sequence)
    except:
        efficacy_score = np.nan

    # ✅ 汇总结果
    result = {
        "toxicity": toxicity_score,
        "stability": stability_score,
        "binding_affinity": affinity_score,
        "bioactivity": efficacy_score
    }

    # 可选：整体评分（简单平均，可替换为模型融合）
    valid_scores = [v for v in result.values() if not np.isnan(v)]
    result["overall_score"] = np.mean(valid_scores) if valid_scores else np.nan

    return result


# 示例调用
if __name__ == "__main__":
    test_seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"
    output = predict_all_properties(test_seq)
    print("Prediction result:\n", output)
