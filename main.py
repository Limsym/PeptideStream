import os
import time
import pandas as pd
from sklearn.model_selection import train_test_split
import re
import torch
import torch.nn as nn
import torch.optim as optim
from models import CharLSTMGenerator
from torch.utils.data import DataLoader
from dataset import load_and_clean_data, save_split, PeptideDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
start_time = time.time()

# === 配置部分 ===
INPUT_FILE = "data/peptides.csv"  # 原始数据路径
OUTPUT_DIR = "data/splits"        # 训练验证集输出路径

if __name__ == "__main__":
    df = load_and_clean_data(INPUT_FILE)
    save_split(df, OUTPUT_DIR)

def load_sequences_from_file(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

# 超参数
seq_length = 20
hidden_dim = 128
num_layers = 1
batch_size = 16 # 26spepoch@16 13spepoch@128 12spepoch@256
epochs = 10
learning_rate = 0.003

# 构建字符映射表
amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
char2idx = {ch: idx for idx, ch in enumerate(amino_acids)}
idx2char = {idx: ch for ch, idx in char2idx.items()}
vocab_size = len(char2idx)

# 加载训练数据
train_path = os.path.join(OUTPUT_DIR, "train.txt")
sequences = load_sequences_from_file(train_path)
train_set = PeptideDataset(sequences, seq_length=seq_length, char2idx=char2idx)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

# 加载验证数据
val_path = os.path.join(OUTPUT_DIR, "val.txt")
sequences = load_sequences_from_file(val_path)
val_set = PeptideDataset(sequences, seq_length=seq_length, char2idx=char2idx)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

# 初始化模型
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
    best_val_loss = checkpoint.get('val_loss_best', float('inf')) # best_val_loss = checkpoint.get('val_loss_best', float('inf')) if os.path.exists(checkpoint_path) else float('inf')
    print(f"从checkpoint {checkpoint_path} 继续训练，起始epoch: {start_epoch}")
    epochs = epochs + start_epoch
else:
    print("未找到checkpoint，训练将从头开始")
    best_val_loss = float('inf')

def validate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            output, _ = model(x)
            loss = criterion(output.view(-1, output.size(-1)), y.view(-1))
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    model.train()
    return avg_loss

def train(model, train_loader, val_loader, optimizer, criterion, start_epoch=0, num_epochs=10, best_val_loss=float('inf')):
    total_train_step = 0
    # 初始化记录结构
    train_losses = []
    val_losses = []
    best_model_state = None
    best_epoch = 0
    patience = 3
    patience_counter = 0

    for epoch in range(start_epoch, num_epochs):
        print(f"📦 正在训练第 {epoch + 1}/{num_epochs} 轮...")
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=True, ncols=100)

        for x, y in progress_bar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output, _ = model(x)
            loss = criterion(output.view(-1, output.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_train_step += 1
            progress_bar.set_postfix(loss=loss.item())

        avg_train_loss = total_loss / len(train_loader)
        writer.add_scalar('train_loss', avg_train_loss, epoch + 1)
        avg_val_loss = validate(model, val_loader, criterion)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")
        writer.add_scalar('val_loss', avg_val_loss, epoch + 1)

        if avg_val_loss <= best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss,
                'val_loss_best': best_val_loss
            }, "best_model.pt")
            print(f"✅ 新最佳模型已保存（Val Loss: {best_val_loss:.4f}）")
        else:
            patience_counter += 1
            print(f"⚠️ 验证集没有提升，EarlyStopping计数器：{patience_counter}/{patience}")
            if patience_counter > patience:
                print("⛔ 提前终止训练！触发 Early Stopping")
                break

        # 保存训练/验证loss历史到csv
        loss_history_df = pd.DataFrame({
            'epoch': list(range(start_epoch + 1, len(train_losses) + start_epoch + 1)),
            'train_loss': train_losses,
            'val_loss': val_losses
        })
        loss_history_df.to_csv("loss_history.csv", index=False)
        print("📁 各轮训练和验证 Loss 已保存至 loss_history.csv\n")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss,
            'val_loss_best': best_val_loss
        }, "checkpoint.pt")

log_dir = os.path.join("logs", f"run-{timestamp}")
writer = SummaryWriter(log_dir=log_dir)
train(model, train_loader, val_loader, optimizer, criterion, start_epoch=start_epoch, num_epochs=epochs, best_val_loss=best_val_loss)
writer.close()
end_time = time.time()
print(f"脚本运行了 {end_time - start_time:.0f} 秒钟")
