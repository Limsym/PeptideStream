import os
import pandas as pd
from sklearn.model_selection import train_test_split
import re

# === 配置部分 ===
INPUT_FILE = "data/peptides.csv"  # 原始数据路径
OUTPUT_DIR = "data/splits"        # 训练验证集输出路径
VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")  # 标准20种氨基酸
MIN_LENGTH = 10  # 你可以调节这个阈值

def is_valid_Sequence(seq):
    """只保留标准氨基酸组成的序列"""
    if not isinstance(seq, str):
        return False
    return all(residue in VALID_AMINO_ACIDS for residue in seq)

def load_and_clean_data(file_path):
    """加载CSV文件并清洗非法序列"""
    df = pd.read_csv(file_path)
    print(f"读取列名: {df.columns.tolist()}")

    df['Sequence'] = df['Sequence'].str.upper()
    df = df[df['Sequence'].apply(is_valid_Sequence)]
    df = df[df['Sequence'].str.len() >= MIN_LENGTH]  # 过滤长度
    return df

def save_split(df, output_dir, train_ratio=0.8):
    """划分训练/验证集并保存为txt文件"""
    # train_df, val_df = train_test_split(df, train_size=train_ratio, random_state=42, stratify=df['label']) 现在没有label
    train_df, val_df = train_test_split(df, train_size=train_ratio, random_state=42)

    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train.txt")
    val_path = os.path.join(output_dir, "val.txt")

    train_df.to_csv(train_path, sep='\t', index=False, header=True)
    val_df.to_csv(val_path, sep='\t', index=False, header=True)

    print(f"✅ 数据已成功拆分并保存：\n - {train_path}\n - {val_path}")

if __name__ == "__main__":
    df = load_and_clean_data(INPUT_FILE)
    print(f"共载入 {len(df)} 条合法多肽序列（长度 ≥ {MIN_LENGTH}）。")
    save_split(df, OUTPUT_DIR)

# 超参数
seq_length = 20
hidden_dim = 128
num_layers = 1
batch_size = 16
epochs = 30
learning_rate = 0.003

# 数据加载
dataset = PeptideDataset(sequences, seq_length=seq_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 模型
model = CharLSTMGenerator(vocab_size, hidden_dim, vocab_size, num_layers)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
