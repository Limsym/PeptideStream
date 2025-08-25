# stability_model_pytorch.py
"""
PyTorch版本的稳定性预测模型
使用MLP + MSE loss进行回归预测
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
from tqdm import tqdm

from embedding_pipeline import EmbeddingPipeline

DATA_CSV = "../data/s2648_clean.csv"
MODEL_PATH = "models/stability_predictor_pytorch.pth"


class StabilityMLP(nn.Module):
    """稳定性预测MLP模型"""
    
    def __init__(self, input_dim=1280, hidden_dims=[512, 256, 128], dropout=0.2):
        super(StabilityMLP, self).__init__()
        
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


def load_data():
    """加载数据"""
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


def train_model(model, train_loader, val_loader, epochs=100, lr=0.001, device="cpu"):
    """训练模型"""
    model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 20
    
    print(f"🏋️ 开始训练，设备: {device}")
    
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
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
            # 保存最佳模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }, MODEL_PATH)
            print(f"✅ 新最佳模型已保存 (Epoch {epoch})")
        else:
            patience_counter += 1
        
        # 打印进度
        if epoch % 10 == 0 or epoch < 10:
            print(f"Epoch {epoch:3d}: Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 早停
        if patience_counter >= patience:
            print(f"⛔ 早停触发 (Epoch {epoch})")
            break
    
    print(f"✅ 训练完成，最佳验证损失: {best_val_loss:.4f}")
    return model


def evaluate_model(model, test_loader, device="cpu"):
    """评估模型"""
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
    
    print(f"📊 评估结果:")
    print(f"   MSE: {mse:.4f}")
    print(f"   R²:  {r2:.4f}")
    
    return mse, r2


def main():
    """主函数"""
    print("🚀 开始PyTorch稳定性预测模型训练...")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 使用设备: {device}")
    
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
    
    for seq, label in tqdm(zip(seqs, labels), total=len(seqs), desc="准备训练数据"):
        if seq in embeddings:
            X.append(embeddings[seq])
            y.append(label)
    
    print(f"✅ 训练数据准备完成，有效样本数: {len(X)}")

    # 数据分割
    print("✂️ 分割训练/验证集...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
    print(f"训练集: {len(X_train)} 样本，验证集: {len(X_val)} 样本")
    
    # 创建数据集和数据加载器
    train_dataset = StabilityDataset(X_train, y_train)
    val_dataset = StabilityDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 创建模型
    print("🏗️ 创建MLP模型...")
    model = StabilityMLP(
        input_dim=1280,
        hidden_dims=[512, 256, 128],
        dropout=0.3
    )
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 模型参数: 总计 {total_params:,}, 可训练 {trainable_params:,}")
    
    # 训练模型
    model = train_model(model, train_loader, val_loader, epochs=200, lr=0.001, device=device)
    
    # 评估模型
    print("📊 最终模型评估...")
    evaluate_model(model, val_loader, device)
    
    print(f"✅ 模型已保存到: {MODEL_PATH}")


if __name__ == "__main__":
    main()
