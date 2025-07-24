import os
import pandas as pd
from sklearn.model_selection import train_test_split
import re
import torch
import torch.nn as nn
import torch.optim as optim
from models import CharLSTMGenerator

# from pytorch_lightning import Trainer
# from pytorch_lightning.callbacks import ModelCheckpoint

from torch.utils.data import DataLoader

from dataset import load_and_clean_data, save_split, PeptideDataset

# === 配置部分 ===
INPUT_FILE = "data/peptides.csv"  # 原始数据路径
OUTPUT_DIR = "data/splits"        # 训练验证集输出路径

if __name__ == "__main__":
    df = load_and_clean_data(INPUT_FILE)
    # print(f"共载入 {len(df)} 条合法多肽序列（长度 ≥ 指定长度）。")
    save_split(df, OUTPUT_DIR)


# 超参数
seq_length = 20
hidden_dim = 128
num_layers = 1
batch_size = 16
epochs = 5
learning_rate = 0.003


def load_sequences_from_file(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

# 加载训练集序列
train_path = os.path.join(OUTPUT_DIR, "train.txt")
sequences = load_sequences_from_file(train_path)
# 构建字符表
chars = sorted(list(set("".join(sequences))))
char2idx = {ch: idx for idx, ch in enumerate(chars)}
idx2char = {idx: ch for ch, idx in char2idx.items()}
vocab_size = len(chars)

# 创建数据集和数据加载器
dataset = PeptideDataset(sequences, seq_length=seq_length, char2idx=char2idx)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 模型
model = CharLSTMGenerator(vocab_size, hidden_dim, vocab_size, num_layers)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


checkpoint_path = "checkpoint.pt"
start_epoch = 0

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"从checkpoint {checkpoint_path} 继续训练，起始epoch: {start_epoch}")
    epochs = epochs + start_epoch
else:
    print("未找到checkpoint，训练将从头开始")

def train(model, dataloader, optimizer, criterion, start_epoch=0, num_epochs=10):
    for epoch in range(start_epoch, num_epochs):
        total_loss = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output, _ = model(x)
            loss = criterion(output.view(-1, output.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss:.4f}")

        # 保存 checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss
        }, "checkpoint.pt")


# 调用训练函数
train(model, dataloader, optimizer, criterion, start_epoch=start_epoch, num_epochs=epochs)
