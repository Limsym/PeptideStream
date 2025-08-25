# stability_model_esm.py
"""
主脚本：训练稳定性预测器，调用批处理 embedding
"""
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm

from embedding_pipeline import EmbeddingPipeline

DATA_CSV = "../data/s2648_clean.csv"  # 修正路径
MODEL_PATH = "models/stability_predictor.pkl"  # 修正路径


def load_data():
    print("📊 正在加载数据...")
    df = pd.read_csv(DATA_CSV)
    seqs = []
    labels = []
    
    # 为数据加载添加进度条
    for _, row in tqdm(df.iterrows(), total=len(df), desc="处理数据行"):
        # 这里简化: 使用 PDB+MUT 拼接为 key (可替换为真实 mutated seq)
        key = f"{row['PDB']}_{row['CHAIN']}_{row['MUT']}"
        seqs.append(key)
        labels.append(row["DDG"])
    
    print(f"✅ 数据加载完成，共 {len(seqs)} 条序列")
    return seqs, labels


def main():
    print("🚀 开始稳定性预测模型训练...")
    
    # 加载数据
    seqs, labels = load_data()
    
    # 生成embeddings
    print("🔬 开始生成ESM2 embeddings...")
    pipeline = EmbeddingPipeline(batch_size=4, device="cpu")
    embeddings = pipeline.embed_sequences(seqs, resume=True)

    # 准备训练数据
    print("📋 准备训练数据...")
    X = []
    y = []
    
    # 为数据准备添加进度条
    for seq, label in tqdm(zip(seqs, labels), total=len(seqs), desc="准备训练数据"):
        if seq in embeddings:
            X.append(embeddings[seq])
            y.append(label)
    
    print(f"✅ 训练数据准备完成，有效样本数: {len(X)}")

    # 数据分割
    print("✂️ 分割训练/验证集...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"训练集: {len(X_train)} 样本，验证集: {len(X_val)} 样本")
    
    # 训练模型
    print("🏋️ 开始训练Ridge回归模型...")
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    print("✅ 模型训练完成")

    # 评估模型
    print("📊 评估模型性能...")
    preds = model.predict(X_val)
    mse = mean_squared_error(y_val, preds)
    r2 = r2_score(y_val, preds)
    print(f"✅ MSE {mse:.4f} | R² {r2:.4f}")

    # 保存模型
    print("💾 保存模型...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"✅ 模型已保存 {MODEL_PATH}")


if __name__ == "__main__":
    main()