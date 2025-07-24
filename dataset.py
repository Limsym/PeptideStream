import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch  # 你后面有用到 torch.zero()


VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")  # 标准20种氨基酸
MIN_LENGTH = 10  # 可以调节这个阈值

def is_valid_Sequence(seq):
    """只保留标准氨基酸组成的序列"""
    if not isinstance(seq, str):
        return False
    return all(residue in VALID_AMINO_ACIDS for residue in seq)

def load_and_clean_data(file_path):
    """加载CSV文件并清洗非法序列"""
    df = pd.read_csv(file_path)
    # print(f"读取列名: {df.columns.tolist()}")

    df['Sequence'] = df['Sequence'].str.upper()
    df = df[df['Sequence'].apply(is_valid_Sequence)]
    df = df[df['Sequence'].str.len() >= MIN_LENGTH]  # 过滤长度
    return df

def save_split(df, output_dir, train_ratio=0.8):
    """划分训练/验证集并保存为txt文件，仅保留sequence字段"""
    # 仅保留sequence列，防止残留其它无用字段
    df = df[['Sequence']].copy()

    # 拆分数据
    # train_df, val_df = train_test_split(df, train_size=train_ratio, random_state=42, stratify=df['label']) 现在没有label
    train_df, val_df = train_test_split(df, train_size=train_ratio, random_state=42)

    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train.txt")
    val_path = os.path.join(output_dir, "val.txt")

    train_df.to_csv(train_path, sep='\t', index=False, header=False)
    val_df.to_csv(val_path, sep='\t', index=False, header=False)

    print(f"✅ 数据已成功拆分并保存：\n - {train_path}\n - {val_path}")

class PeptideDataset(Dataset):
    def __init__(self, sequences, seq_length=20, char2idx=None):
        self.data = []
        self.seq_length = seq_length
        self.char2idx = char2idx
        self.vocab_size = len(char2idx)

        for seq in sequences:
            for i in range(len(seq) - seq_length):
                input_seq = seq[i:i+seq_length]
                target_seq = seq[i+1:i+seq_length+1]
                self.data.append((input_seq, target_seq))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_seq, target_seq = self.data[idx]
        x = torch.zeros(self.seq_length, self.vocab_size)
        y = torch.zeros(self.seq_length, dtype=torch.long)
        for i, ch in enumerate(input_seq):
            x[i][self.char2idx[ch]] = 1.0
        for i, ch in enumerate(target_seq):
            y[i] = self.char2idx[ch]
        return x, y

