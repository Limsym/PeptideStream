import os
import pandas as pd
from sklearn.model_selection import train_test_split
import re

from torch.utils.data import DataLoader

from dataset import load_and_clean_data, save_split, PeptideDataset

# === 配置部分 ===
INPUT_FILE = "data/peptides.csv"  # 原始数据路径
OUTPUT_DIR = "data/splits"        # 训练验证集输出路径

if __name__ == "__main__":
    df = load_and_clean_data(INPUT_FILE)
    print(f"共载入 {len(df)} 条合法多肽序列（长度 ≥ 指定长度）。")
    save_split(df, OUTPUT_DIR)


# 超参数
seq_length = 20
hidden_dim = 128
num_layers = 1
batch_size = 16
epochs = 30
learning_rate = 0.003


def load_sequences_from_file(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

# 加载训练集序列
train_path = os.path.join(OUTPUT_DIR, "train.txt")
sequences = load_sequences_from_file(train_path)

# 创建数据集和数据加载器
dataset = PeptideDataset(sequences, seq_length=seq_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 模型
model = CharLSTMGenerator(vocab_size, hidden_dim, vocab_size, num_layers)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
