# embedding_pipeline.py
"""
批处理 + 断点续训支持的 embedding 提取
"""
import os
import torch
import pickle
import pandas as pd
from transformers import EsmModel, EsmTokenizer
from tqdm import tqdm

CHECKPOINT = "facebook/esm2_t33_650M_UR50D"
SAVE_FILE = "embeddings_cache.pkl"

class EmbeddingPipeline:
    def __init__(self, checkpoint=CHECKPOINT, batch_size=8, device="cpu"):
        self.tokenizer = EsmTokenizer.from_pretrained(checkpoint)
        self.model = EsmModel.from_pretrained(checkpoint)
        self.model.to(device)
        self.model.eval()
        self.batch_size = batch_size
        self.device = device

    def embed_sequences(self, sequences, resume=True):
        results = {}
        if resume and os.path.exists(SAVE_FILE):
            with open(SAVE_FILE, "rb") as f:
                results = pickle.load(f)
            print(f"已加载缓存: {len(results)} 条")

        start = len(results)
        # 计算需要处理的批次数量
        total_batches = (len(sequences) - start + self.batch_size - 1) // self.batch_size
        
        # 使用tqdm创建进度条
        with tqdm(total=total_batches, desc="生成ESM2 embeddings", unit="batch") as pbar:
            for i in range(start, len(sequences), self.batch_size):
                batch = sequences[i : i + self.batch_size]
                inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(self.device)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    emb = outputs.last_hidden_state.mean(dim=1).cpu()
                for seq, vec in zip(batch, emb):
                    results[seq] = vec.numpy()
                
                # 更新进度条
                pbar.update(1)
                pbar.set_postfix({
                    'processed': f"{min(i + self.batch_size, len(sequences))}/{len(sequences)}",
                    'cached': len(results)
                })
                
                if (i // self.batch_size) % 10 == 0:
                    with open(SAVE_FILE, "wb") as f:
                        pickle.dump(results, f)
                    pbar.set_postfix({
                        'processed': f"{min(i + self.batch_size, len(sequences))}/{len(sequences)}",
                        'cached': len(results),
                        'saved': '✓'
                    })
        
        with open(SAVE_FILE, "wb") as f:
            pickle.dump(results, f)
        print(f"✅ 完成！共生成 {len(results)} 个embeddings")
        return results