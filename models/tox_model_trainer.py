# tox_model_trainer.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.ensemble import GradientBoostingClassifier  # 可替换为 LGBMClassifier、MLP 等
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ========== 1. 加载数据 ==========
# 假设你已经保存好 .npy 和 .csv 文件
embedding_path = "../data/esm2_toxin_embeddings.npy"
label_csv_path = "../data/ToxinPred.csv"

X = np.load("../data/esm2_toxin_embeddings.npy", allow_pickle=True)

# 加载标签，要求有 'Sequence', 'Positive' 列
df = pd.read_csv("../data/ToxinPred.csv")
df = df[["Sequence", "Positive"]].dropna()
valid_aas = set("ACDEFGHIKLMNPQRSTVWY")
df = df[df["Sequence"].apply(lambda s: all(aa in valid_aas for aa in s))].drop_duplicates()
y = df["Positive"].astype(int).values

# ========== 2. 检查维度一致性 ==========
assert len(X) == len(y), f"Mismatch: embeddings({len(X)}), labels({len(y)})"

# ========== 3. 拆分训练/验证集 ==========
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ========== 4. 构建分类器 ==========
model = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# ========== 5. 评估 ==========
y_pred = model.predict(X_val)
y_proba = model.predict_proba(X_val)[:, 1]

acc = accuracy_score(y_val, y_pred)
auc = roc_auc_score(y_val, y_proba)

print(f"\n✅ Accuracy: {acc:.4f} | ROC-AUC: {auc:.4f}\n")
print("Classification Report:\n", classification_report(y_val, y_pred))
print("Confusion Matrix:")
cm = confusion_matrix(y_val, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# ========== 6. 保存模型 ==========
joblib.dump(model, "../models/toxicity_predictor.pkl")
print("✅ 模型已保存为 'toxicity_predictor.pkl'")
