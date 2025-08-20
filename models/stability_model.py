# stability_model.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import os

# ========== 1. 加载数据 ==========
data_path = "/data/S2648.csv"
df = pd.read_csv(data_path, sep=",")  # 如果是逗号分隔改成 sep=","
df = df[["PDB", "CHAIN", "MUT", "DDG"]].dropna()

# ========== 2. 标签 ==========
y = df["DDG"].values  # 回归任务

# ========== 3. 模拟 embedding（先占位，后续替换为真实 ESM2） ==========
np.random.seed(42)
X = np.random.rand(len(df), 1280)  # 1280 为 ESM2 embedding 维度

# ========== 4. 拆分训练/验证集 ==========
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ========== 5. 训练回归模型 ==========
model = GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# ========== 6. 评估 ==========
y_pred = model.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print(f"✅ MSE: {mse:.4f} | R²: {r2:.4f}")

# ========== 7. 保存模型 ==========
os.makedirs("models", exist_ok=True)
joblib.dump(model, "/models/stability_predictor.pkl")
print("✅ 模型已保存为 '/models/stability_predictor.pkl'")
