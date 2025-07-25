import os
import time
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
from torch.utils.tensorboard import SummaryWriter


# === 配置部分 ===
INPUT_FILE = "data/peptides.csv"  # 原始数据路径
OUTPUT_DIR = "data/splits"        # 训练验证集输出路径

if __name__ == "__main__":
    df = load_and_clean_data(INPUT_FILE)
    # print(f"共载入 {len(df)} 条合法多肽序列（长度 ≥ 指定长度）。")
    save_split(df, OUTPUT_DIR)

def load_sequences_from_file(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


# 超参数
seq_length = 20
hidden_dim = 128
num_layers = 1
batch_size = 16
epochs = 1
learning_rate = 0.003
# 记录测试的次数
total_test_step = 0
# 训练集：加载训练集序列
train_path = os.path.join(OUTPUT_DIR, "train.txt")
sequences = load_sequences_from_file(train_path)

# 构建字符表
chars = sorted(list(set("".join(sequences))))
char2idx = {ch: idx for idx, ch in enumerate(chars)}
idx2char = {idx: ch for ch, idx in char2idx.items()}
vocab_size = len(chars)

# 创建数据集和数据加载器
train_set = PeptideDataset(sequences, seq_length=seq_length, char2idx=char2idx)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

# 验证集：加载训练集序列
val_path = os.path.join(OUTPUT_DIR, "val.txt")
sequences = load_sequences_from_file(val_path)
val_set = PeptideDataset(sequences, seq_length=seq_length, char2idx=char2idx)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)


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

def validate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            output, _ = model(x)
            output = output.view(-1, output.size(-1))
            y = y.view(-1)
            loss = criterion(output, y)
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    print(f"验证集平均Loss: {avg_loss:.4f}")
    model.train()

def train(model, train_loader, val_loader, optimizer, criterion, start_epoch=0, num_epochs=10):
    # 记录训练的次数
    total_train_step = 0
    for epoch in range(start_epoch, num_epochs):
        print("-------第{}轮训练开始-------".format(epoch + 1))
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output, _ = model(x)
            loss = criterion(output.view(-1, output.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            total_train_step += 1

            if total_train_step % 250 == 0:
                end_time = time.time()
                print(end_time - start_time)
                print("训练次数：{}, Loss：{}".format(total_train_step, loss.item()))

            writer.add_scalar('train_loss', loss.item(), total_train_step)

        # 每轮结束时打印一次总loss
        avg_epoch_loss = total_loss / len(train_loader)
        print(f"✅ Epoch [{epoch + 1}/{num_epochs}] 完成，平均Loss: {avg_epoch_loss:.4f}")

        # 验证步骤开始
        validate(model, val_loader, criterion)

        # 保存 checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss
        }, "checkpoint.pt")

writer = SummaryWriter(log_dir='runs/')

start_time = time.time()
# 调用训练函数
train(model, train_loader, val_loader, optimizer, criterion, start_epoch=start_epoch, num_epochs=epochs)

writer.close()
# tensorboard --logdir=runs
