# pip install transformers
# Xet Storage is enabled for this repo, but the 'hf_xet' package is not installed. Falling back to regular HTTP download. For better performance, install the package with: `pip install huggingface_hub[hf_xet]` or `pip install hf_xet`
# bash
# hf download facebook/esm2_t33_650M_UR50D

from transformers import EsmTokenizer, EsmModel
import torch
from esm_loader import load_esm_model
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Union

local_path = "d:/Users/Siqiniq/.cache/huggingface/hub/models--facebook--esm2_t33_650M_UR50D/snapshots/08e4846e537177426273712802403f7ba8261b6c/"
tokenizer = EsmTokenizer.from_pretrained(local_path)
model = EsmModel.from_pretrained(local_path)
model.eval()

# 测试
# sequence = "MKTFFVLVVLILALVG"
# inputs = tokenizer(sequence, return_tensors="pt")
# with torch.no_grad():
#     outputs = model(**inputs)
# embedding = outputs.last_hidden_state  # shape: [1, L, D]
#
# # 获取最后一层隐藏状态 (B, L, D)
# token_embeddings = outputs.last_hidden_state
#
# print("Embedding shape:", token_embeddings.shape)  # 示例输出: torch.Size([1, L, D])

def load_esm_components(local_path):
    """加载ESM2模型和tokenizer"""
    print(f"📥 加载ESM2模型从: {local_path}")
    tokenizer = EsmTokenizer.from_pretrained(local_path)
    model = EsmModel.from_pretrained(local_path)
    model.eval()
    print("✅ ESM2模型加载完成")
    return model, tokenizer

def extract_protein_embedding(sequence: str, model, tokenizer):
    """提取单个蛋白质序列的embedding"""
    inputs = tokenizer(sequence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    # 通常使用第一个token对应的向量（即CLS）作为序列表示
    cls_embedding = outputs.last_hidden_state[0, 0].numpy()
    return cls_embedding

def extract_protein_embeddings_batch(sequences: List[str], model, tokenizer, 
                                   batch_size: int = 8, show_progress: bool = True) -> np.ndarray:
    """
    批量提取蛋白质序列的embeddings，带进度条
    
    Args:
        sequences: 蛋白质序列列表
        model: ESM2模型
        tokenizer: ESM2 tokenizer
        batch_size: 批处理大小
        show_progress: 是否显示进度条
    
    Returns:
        embeddings: numpy数组，形状为 (n_sequences, embedding_dim)
    """
    print(f"🔬 开始批量提取embeddings，共 {len(sequences)} 条序列")
    
    embeddings = []
    total_batches = (len(sequences) + batch_size - 1) // batch_size
    
    if show_progress:
        pbar = tqdm(total=total_batches, desc="提取embeddings", unit="batch")
    
    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i:i + batch_size]
        
        # 处理批次
        batch_embeddings = []
        for seq in batch_seqs:
            emb = extract_protein_embedding(seq, model, tokenizer)
            batch_embeddings.append(emb)
        
        embeddings.extend(batch_embeddings)
        
        if show_progress:
            pbar.update(1)
            pbar.set_postfix({
                'processed': f"{min(i + batch_size, len(sequences))}/{len(sequences)}",
                'batch_size': len(batch_seqs)
            })
    
    if show_progress:
        pbar.close()
    
    print(f"✅ 完成！共提取 {len(embeddings)} 个embeddings")
    return np.array(embeddings)

def save_embeddings(embeddings: np.ndarray, sequences: List[str], 
                   npy_path: str = "protein_embeddings.npy", 
                   csv_path: str = "protein_embeddings.csv"):
    """
    保存embeddings到文件
    
    Args:
        embeddings: embedding数组
        sequences: 对应的序列列表
        npy_path: .npy文件保存路径
        csv_path: .csv文件保存路径
    """
    print("💾 保存embeddings...")
    
    # 保存为 .npy
    np.save(npy_path, embeddings)
    print(f"✅ embeddings已保存为: {npy_path}")
    
    # 保存为 .csv（带上序列索引）
    df = pd.DataFrame(embeddings, index=sequences)
    df.to_csv(csv_path)
    print(f"✅ embeddings已保存为: {csv_path}")


# 示例使用
if __name__ == "__main__":
    # 测试序列
    test_sequences = [
        "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ", 
        "GSHSMRYFYTAMSRPGRGEPRFISVGYVDDTQFVRF",
        "MKTFFVLVVLILALVG"
    ]
    
    print("🧪 测试批量embedding提取...")
    
    # 加载模型
    model, tokenizer = load_esm_components(local_path)
    
    # 批量提取embeddings
    embeddings = extract_protein_embeddings_batch(test_sequences, model, tokenizer, batch_size=2)
    
    # 保存结果
    save_embeddings(embeddings, test_sequences)
    
    print(f"🎉 测试完成！embeddings形状: {embeddings.shape}")