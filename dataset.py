class PeptideDataset(Dataset):
    def __init__(self, sequences, seq_length=20):
        self.data = []
        for seq in sequences:
            for i in range(len(seq) - seq_length):
                input_seq = seq[i:i+seq_length]
                target_seq = seq[i+1:i+seq_length+1]
                self.data.append((input_seq, target_seq))
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_seq, target_seq = self.data[idx]
        x = torch.zeros(self.seq_length, vocab_size)
        y = torch.zeros(self.seq_length, dtype=torch.long)
        for i, ch in enumerate(input_seq):
            x[i][char2idx[ch]] = 1.0
        for i, ch in enumerate(target_seq):
            y[i] = char2idx[ch]
        return x, y
