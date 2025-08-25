# tox_model_trainer.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.ensemble import GradientBoostingClassifier  # 可替换为 LGBMClassifier、MLP 等
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ========== 1. 加载数据 ==========
print("📊 开始加载数据...")
# 假设你已经保存好 .npy 和 .csv 文件
embedding_path = "../data/esm2_toxin_embeddings.npy"
label_csv_path = "../data/ToxinPred.csv"

print("🔬 加载ESM2 embeddings...")
X = np.load("../data/esm2_toxin_embeddings.npy", allow_pickle=True)
print(f"✅ embeddings加载完成，形状: {X.shape}")

# 加载标签，要求有 'Sequence', 'Positive' 列
print("📋 加载标签数据...")
df = pd.read_csv("../data/ToxinPred.csv")
df = df[["Sequence", "Positive"]].dropna()
print(f"原始数据: {len(df)} 条")

# 数据清洗
print("🧹 清洗数据...")
valid_aas = set("ACDEFGHIKLMNPQRSTVWY")
df = df[df["Sequence"].apply(lambda s: all(aa in valid_aas for aa in s))].drop_duplicates()
print(f"清洗后数据: {len(df)} 条")

y = df["Positive"].astype(int).values

# ========== 2. 检查维度一致性 ==========
print("🔍 检查数据一致性...")
assert len(X) == len(y), f"Mismatch: embeddings({len(X)}), labels({len(y)})"
print("✅ 数据维度一致")

# ========== 3. 拆分训练/验证集 ==========
print("✂️ 分割训练/验证集...")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"训练集: {len(X_train)} 样本，验证集: {len(X_val)} 样本")

# ========== 4. 构建分类器 ==========
print("🏋️ 开始训练GradientBoosting分类器...")
model = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)

# 为训练过程添加进度条（通过verbose参数）
model.fit(X_train, y_train)
print("✅ 模型训练完成")

# ========== 5. 评估 ==========
print("📊 评估模型性能...")
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
print("💾 保存模型...")
joblib.dump(model, "../models/toxicity_predictor.pkl")
print("✅ 模型已保存为 'toxicity_predictor.pkl'")
