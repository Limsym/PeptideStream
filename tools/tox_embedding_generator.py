# tox_embedding_generator.py

import pandas as pd
import numpy as np
import os
import time
from tqdm import tqdm
from embedding.esm_embedding_extractor import load_esm_components, extract_protein_embedding


def generate_embeddings_in_batches(df, model, tokenizer,
                                   seq_col='Sequence', label_col='Positive',
                                   save_path='embeddings_partial.npy',
                                   save_every=50):
    """
    分批提取 embeddings，带进度显示，支持中断续训。
    """
    print("🔬 开始生成ESM2 embeddings...")
    
    # 转换标签为 int（True -> 1, False -> 0）
    print("📋 预处理数据...")
    df = df[[seq_col, label_col]].dropna()
    valid_aas = set("ACDEFGHIKLMNPQRSTVWY")
    df = df[df[seq_col].apply(lambda s: all(aa in valid_aas for aa in s))].drop_duplicates()
    df['Label'] = df[label_col].astype(int)

    sequences = df[seq_col].tolist()
    labels = df['Label'].values
    
    print(f"✅ 数据预处理完成，有效序列数: {len(sequences)}")

    # 加载已存在的 embedding 缓存
    if os.path.exists(save_path):
        existing = list(np.load(save_path, allow_pickle=True))
        start_idx = len(existing)
        embeddings = existing
        print(f"📁 已加载 {start_idx} 个 embeddings，继续从第 {start_idx + 1} 条")
    else:
        embeddings = []
        start_idx = 0
        print(f"🆕 开始全新生成 embeddings")

    # 主循环，带进度条
    batch_start_time = time.time()
    
    # 使用tqdm创建更详细的进度条
    with tqdm(range(start_idx, len(sequences)), 
              desc="生成ESM2 embeddings", 
              unit="seq",
              ncols=100,
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
        
        for i in pbar:
            seq = sequences[i]
            emb = extract_protein_embedding(seq, model, tokenizer)
            embeddings.append(emb)

            # 更新进度条信息
            pbar.set_postfix({
                'processed': f"{i+1}/{len(sequences)}",
                'cached': len(embeddings),
                'eta': f"{pbar.eta:.0f}s"
            })

            # 每到一个保存点，保存并打印该批次时间
            if (i + 1) % save_every == 0 or (i + 1) == len(sequences):
                np.save(save_path, np.array(embeddings, dtype=object))
                batch_time = time.time() - batch_start_time
                pbar.set_postfix({
                    'processed': f"{i + 1}/{len(sequences)}",
                    'cached': len(embeddings),
                    'saved': '✓',
                    'batch_time': f"{batch_time:.1f}s"
                })
                print(f"\n💾 已保存至 {save_path}，共 {i + 1} 个，本批耗时 {batch_time:.2f}s")
                batch_start_time = time.time()  # 重置时间戳

    print(f"✅ 完成！共生成 {len(embeddings)} 个embeddings")
    return np.array(embeddings), labels


# 主执行部分
if __name__ == "__main__":
    print("🚀 开始毒性预测embedding生成...")
    
    # 加载模型
    print("📥 加载ESM2模型...")
    esm_path = "d:/Users/Siqiniq/.cache/huggingface/hub/models--facebook--esm2_t33_650M_UR50D/snapshots/08e4846e537177426273712802403f7ba8261b6c/"
    model, tokenizer = load_esm_components(esm_path)
    print("✅ 模型加载完成")

    # 读取数据
    print("📊 读取数据文件...")
    df = pd.read_csv("../data/ToxinPred.csv")
    print(f"✅ 数据读取完成，共 {len(df)} 条记录")

    # 执行增强版 embedding 提取
    X, y = generate_embeddings_in_batches(
        df,
        model,
        tokenizer,
        save_path="../data/esm2_toxin_embeddings.npy",  # 自定义保存路径
        save_every=50                                 # 每处理50条保存一次
    )
    
    print(f"🎉 全部完成！最终embeddings形状: {X.shape}")