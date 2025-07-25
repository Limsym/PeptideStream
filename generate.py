import torch
import torch.nn.functional as F
from models import CharLSTMGenerator

def generate_sequence(model, start_str, char2idx, idx2char, max_length=50, temperature=1.0):
    model.eval()
    input_eval = torch.zeros(1, len(start_str), len(char2idx)).to(device)
    for t, ch in enumerate(start_str):
        input_eval[0, t, char2idx[ch]] = 1.0
        hidden = None
    generated = start_str

    for _ in range(max_length - len(start_str)):
        output, hidden = model(input_eval, hidden)
        output = output[:, -1, :]  # 只取最后一个时间步的输出
        probs = F.softmax(output / temperature, dim=-1).squeeze()
        char_id = torch.multinomial(probs, 1).item()

        next_char = idx2char[char_id]
        generated += next_char

        input_eval = torch.zeros(1, 1, len(char2idx)).to(device)
        input_eval[0, 0, char_id] = 1.0

    return generated

def generate_batch(model, n, start_str, char2idx, idx2char, max_length=50, temperature=1.0):
    return [generate_sequence(model, start_str, char2idx, idx2char, max_length, temperature)
            for _ in range(n)]

# 加载 checkpoint
vocab = list("ACDEFGHIKLMNPQRSTVWY")  # 20种标准氨基酸
char2idx = {ch: idx for idx, ch in enumerate(vocab)}
idx2char = {idx: ch for ch, idx in char2idx.items()}
vocab_size = len(vocab)
hidden_dim = 128
num_layers = 1

model = CharLSTMGenerator(vocab_size, hidden_dim, vocab_size, num_layers)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load("checkpoint.pt", map_location=device)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  # 设置为评估模式

# 示例：生成 10 个多肽序列
generated_peptides = generate_batch(model, n=10, start_str='M',  # 假设多肽从'M'起始
                                    char2idx=char2idx, idx2char=idx2char,
                                    max_length=50, temperature=0.8)

for i, pep in enumerate(generated_peptides, 1):
    print(f"[{i}] {pep}")
