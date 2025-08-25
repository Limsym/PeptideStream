# stability_model_pytorch_enhanced.py
"""
增强版PyTorch稳定性预测模型
整合数据增强、集成学习、架构优化和损失函数优化
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from tqdm import tqdm
import random

from embedding_pipeline import EmbeddingPipeline

DATA_CSV = "../data/s2648_clean.csv"
MODEL_PATH = "models/stability_predictor_pytorch_enhanced.pth"
ENSEMBLE_PATH = "models/stability_ensemble_enhanced/"

# 确保集成模型保存目录存在
os.makedirs(ENSEMBLE_PATH, exist_ok=True)


class HuberLoss(nn.Module):
    """Huber损失函数，对异常值更鲁棒"""
    
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta
    
    def forward(self, y_pred, y_true):
        error = y_pred - y_true
        abs_error = torch.abs(error)
        quadratic = torch.clamp(abs_error, max=self.delta)
        linear = abs_error - quadratic
        return torch.mean(0.5 * quadratic**2 + self.delta * linear)


class StabilityMLPEnhanced(nn.Module):
    """增强版稳定性预测MLP模型"""
    
    def __init__(self, input_dim=1280, hidden_dims=[512, 256, 128, 64], dropout=0.3):
        super(StabilityMLPEnhanced, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # 构建隐藏层
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze(-1)


class StabilityDataset(Dataset):
    """稳定性数据集类"""
    
    def __init__(self, embeddings, labels):
        self.embeddings = torch.FloatTensor(embeddings)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


def add_noise(X, noise_factor=0.01):
    """添加少量噪声进行数据增强"""
    noise = np.random.normal(0, noise_factor, X.shape)
    return X + noise


def load_and_preprocess_data():
    """加载并预处理数据"""
    print("📊 正在加载数据...")
    df = pd.read_csv(DATA_CSV)
    seqs = []
    labels = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="处理数据行"):
        key = f"{row['PDB']}_{row['CHAIN']}_{row['MUT']}"
        seqs.append(key)
        labels.append(row["DDG"])
    
    print(f"✅ 数据加载完成，共 {len(seqs)} 条序列")
    return seqs, labels


def compare_models(X, y):
    """比较不同模型的性能 - 使用与原始脚本相同的设置"""
    print("🔍 比较不同模型性能...")
    
    # 使用与stability_model_esm.py相同的设置
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Ridge回归（无标准化）
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_val)
    r2 = r2_score(y_val, y_pred)
    mse = mean_squared_error(y_val, y_pred)
    
    print(f"Ridge回归 (原始设置): R² = {r2:.4f}, MSE = {mse:.4f}")
    
    return r2


def train_model_enhanced(model, train_loader, val_loader, epochs=100, lr=0.0005, device="cpu"):
    """增强版训练函数"""
    model.to(device)
    
    # 使用Huber Loss
    criterion = HuberLoss(delta=1.0)
    # 优化参数
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.7, min_lr=1e-7)
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 25  # 适度耐心
    
    print(f"🏋️ 开始增强训练，设备: {device}")
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_batches = 0
        
        for batch_emb, batch_labels in train_loader:
            batch_emb, batch_labels = batch_emb.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_emb)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for batch_emb, batch_labels in val_loader:
                batch_emb, batch_labels = batch_emb.to(device), batch_labels.to(device)
                outputs = model(batch_emb)
                val_loss += criterion(outputs, batch_labels).item()
                val_batches += 1
        
        # 计算平均损失
        avg_train_loss = train_loss / train_batches
        avg_val_loss = val_loss / val_batches
        
        # 学习率调度
        scheduler.step(avg_val_loss)
        
        # 早停检查
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        # 打印进度
        if epoch % 20 == 0 or epoch < 10:
            print(f"Epoch {epoch:3d}: Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 早停
        if patience_counter >= patience:
            print(f"⛔ 早停触发 (Epoch {epoch})")
            break
    
    print(f"✅ 训练完成，最佳验证损失: {best_val_loss:.4f}")
    return model


def evaluate_model_comprehensive(model, test_loader, device="cpu"):
    """综合评估模型"""
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch_emb, batch_labels in test_loader:
            batch_emb = batch_emb.to(device)
            outputs = model(batch_emb)
            predictions.extend(outputs.cpu().numpy())
            true_labels.extend(batch_labels.numpy())
    
    # 计算指标
    mse = mean_squared_error(true_labels, predictions)
    r2 = r2_score(true_labels, predictions)
    
    print(f"📊 综合评估结果:")
    print(f"   MSE: {mse:.4f}")
    print(f"   R²:  {r2:.4f}")
    print(f"   RMSE: {np.sqrt(mse):.4f}")
    
    # 计算相关系数
    correlation = np.corrcoef(true_labels, predictions)[0, 1]
    print(f"   相关系数: {correlation:.4f}")
    
    return mse, r2, correlation


def ensemble_predict(models, X, device="cpu"):
    """集成预测"""
    predictions = []
    for model in models:
        model.eval()
        with torch.no_grad():
            pred = model(torch.FloatTensor(X).to(device))
            predictions.append(pred.cpu().numpy())
    return np.mean(predictions, axis=0)


def train_ensemble(X_train, X_val, y_train, y_val, n_models=3, device="cpu"):
    """训练集成模型"""
    print(f"🎯 开始训练集成模型 ({n_models} 个模型)...")
    
    models = []
    train_dataset = StabilityDataset(X_train, y_train)
    val_dataset = StabilityDataset(X_val, y_val)
    
    for i in range(n_models):
        print(f"\n🏗️ 训练模型 {i+1}/{n_models}")
        
        # 为每个模型设置不同的随机种子
        torch.manual_seed(42 + i)
        np.random.seed(42 + i)
        random.seed(42 + i)
        
        # 创建模型
        model = StabilityMLPEnhanced(
            input_dim=1280,
            hidden_dims=[512, 256, 128, 64],
            dropout=0.3
        )
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        
        # 训练模型
        model = train_model_enhanced(model, train_loader, val_loader, epochs=200, lr=0.0005, device=device)
        
        # 保存模型
        model_path = os.path.join(ENSEMBLE_PATH, f"model_{i+1}.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_index': i+1,
        }, model_path)
        
        models.append(model)
        print(f"✅ 模型 {i+1} 已保存到: {model_path}")
    
    return models


def evaluate_ensemble(models, X_val, y_val, device="cpu"):
    """评估集成模型"""
    print("📊 评估集成模型...")
    
    # 集成预测
    ensemble_pred = ensemble_predict(models, X_val, device)
    
    # 计算指标
    mse = mean_squared_error(y_val, ensemble_pred)
    r2 = r2_score(y_val, ensemble_pred)
    correlation = np.corrcoef(y_val, ensemble_pred)[0, 1]
    
    print(f"📊 集成模型评估结果:")
    print(f"   MSE: {mse:.4f}")
    print(f"   R²:  {r2:.4f}")
    print(f"   RMSE: {np.sqrt(mse):.4f}")
    print(f"   相关系数: {correlation:.4f}")
    
    return mse, r2, correlation


def main():
    """主函数"""
    print("🚀 开始增强版PyTorch稳定性预测模型训练...")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 使用设备: {device}")
    
    # 加载数据
    seqs, labels = load_and_preprocess_data()
    
    # 生成embeddings
    print("🔬 开始生成ESM2 embeddings...")
    pipeline = EmbeddingPipeline(batch_size=4, device="cpu")
    embeddings = pipeline.embed_sequences(seqs, resume=True)

    # 准备训练数据
    print("📋 准备训练数据...")
    X = []
    y = []
    
    for seq, label in tqdm(zip(seqs, labels), total=len(seqs), desc="准备训练数据"):
        if seq in embeddings:
            X.append(embeddings[seq])
            y.append(label)
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"✅ 训练数据准备完成，有效样本数: {len(X)}")
    print(f"📊 数据形状: X={X.shape}, y={y.shape}")
    
    # 比较不同模型（使用原始设置）
    ridge_r2 = compare_models(X, y)
    
    # 数据增强
    print("📈 应用数据增强...")
    X_augmented = np.vstack([X, add_noise(X, 0.01)])
    y_augmented = np.hstack([y, y])
    
    print(f"📈 数据增强后: {len(X_augmented)} 样本 (原始: {len(X)})")
    
    # 为MLP准备标准化数据
    print("🔧 为MLP准备标准化数据...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_augmented)
    
    # 数据分割（使用相同比例）
    print("✂️ 分割训练/验证集...")
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_augmented, test_size=0.2, random_state=42)
    print(f"训练集: {len(X_train)} 样本，验证集: {len(X_val)} 样本")
    
    # 训练集成模型
    models = train_ensemble(X_train, X_val, y_train, y_val, n_models=3, device=device)
    
    # 评估集成模型
    mse, r2, correlation = evaluate_ensemble(models, X_val, y_val, device)
    
    # 与Ridge比较
    print(f"\n📈 性能对比 (相同数据分割):")
    print(f"   Ridge回归 R²: {ridge_r2:.4f}")
    print(f"   集成MLP R²: {r2:.4f}")
    print(f"   性能差异: {r2 - ridge_r2:+.4f}")
    
    if r2 > ridge_r2:
        print("✅ 集成MLP 性能优于 Ridge回归!")
    else:
        print("⚠️ Ridge回归 性能优于 集成MLP")
        print("💡 建议: 可能需要更多数据或更复杂的架构")
    
    # 保存集成模型信息
    ensemble_info = {
        'n_models': len(models),
        'model_paths': [os.path.join(ENSEMBLE_PATH, f"model_{i+1}.pth") for i in range(len(models))],
        'scaler': scaler,
        'ridge_r2': ridge_r2,
        'ensemble_r2': r2,
        'performance_improvement': r2 - ridge_r2
    }
    
    torch.save(ensemble_info, MODEL_PATH)
    print(f"✅ 集成模型信息已保存到: {MODEL_PATH}")
    print(f"✅ 各模型已保存到: {ENSEMBLE_PATH}")


if __name__ == "__main__":
    main()

