# pip install transformers
# Xet Storage is enabled for this repo, but the 'hf_xet' package is not installed. Falling back to regular HTTP download. For better performance, install the package with: `pip install huggingface_hub[hf_xet]` or `pip install hf_xet`
# pip install huggingface_hub[hf_xet]
# bash
# hf download facebook/esm2_t33_650M_UR50D

from transformers import EsmTokenizer, EsmModel
import torch
from esm_loader import load_esm_model
local_path = "d:/Users/Siqiniq/.cache/huggingface/hub/models--facebook--esm2_t33_650M_UR50D/snapshots/08e4846e537177426273712802403f7ba8261b6c/"
tokenizer = EsmTokenizer.from_pretrained(local_path)
model = EsmModel.from_pretrained(local_path)
model.eval()

sequence = "MKTFFVLVVLILALVG"
inputs = tokenizer(sequence, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
embedding = outputs.last_hidden_state  # shape: [1, L, D]

# 获取最后一层隐藏状态 (B, L, D)
token_embeddings = outputs.last_hidden_state

print("Embedding shape:", token_embeddings.shape)  # 示例输出: torch.Size([1, L, D])

import numpy as np
import pandas as pd

def extract_protein_embedding(sequence: str, model, tokenizer):
    inputs = tokenizer(sequence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    # 通常使用第一个token对应的向量（即CLS）作为序列表示
    cls_embedding = outputs.last_hidden_state[0, 0].numpy()
    return cls_embedding

sequences = ["MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ", "GSHSMRYFYTAMSRPGRGEPRFISVGYVDDTQFVRF"]
embeddings = [extract_protein_embedding(seq, model, tokenizer) for seq in sequences]
embeddings = np.array(embeddings)

# 保存为 .npy
np.save("protein_embeddings.npy", embeddings)

# 保存为 .csv（带上序列索引）
pd.DataFrame(embeddings, index=sequences).to_csv("protein_embeddings.csv")