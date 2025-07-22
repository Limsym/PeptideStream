import pandas as pd

def load_sequences_from_file(filepath):
    df = pd.read_csv(filepath, sep='\t')
    df = df.dropna(subset=['Sequence'])  # 丢掉无效行
    sequences = df['Sequence'].astype(str).tolist()
    return sequences

from torch.utils.data import Dataset
import torch

amino_acids = "ACDEFGHIKLMNPQRSTVWY"
aa_to_idx = {aa: idx+1 for idx, aa in enumerate(amino_acids)}  # 0 为 padding

class PeptideDataset(Dataset):
    def __init__(self, sequences, max_length=100):
        self.sequences = sequences
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        seq_idx = [aa_to_idx.get(aa, 0) for aa in seq]  # 未知字符记为0
        if len(seq_idx) < self.max_length:
            seq_idx += [0] * (self.max_length - len(seq_idx))  # padding
        else:
            seq_idx = seq_idx[:self.max_length]
        return torch.tensor(seq_idx, dtype=torch.long)


from torch.utils.data import DataLoader

# 加载数据
sequences = load_sequences_from_file("splits/train.txt")
dataset = PeptideDataset(sequences)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
