import pandas as pd
import numpy as np
import os
import time
from tqdm import tqdm
from esm_embedding_extractor import load_esm_components, extract_protein_embedding


def generate_embeddings_in_batches(df, model, tokenizer,
                                   seq_col='Sequence', label_col='Positive',
                                   save_path='embeddings_partial.npy',
                                   save_every=50):
    """
    分批提取 embeddings，带进度显示，支持中断续训。
    """
    # 转换标签为 int（True -> 1, False -> 0）
    df = df[[seq_col, label_col]].dropna()
    valid_aas = set("ACDEFGHIKLMNPQRSTVWY")
    df = df[df[seq_col].apply(lambda s: all(aa in valid_aas for aa in s))].drop_duplicates()
    df['Label'] = df[label_col].astype(int)

    sequences = df[seq_col].tolist()
    labels = df['Label'].values

    # 加载已存在的 embedding 缓存
    if os.path.exists(save_path):
        existing = list(np.load(save_path, allow_pickle=True))
        start_idx = len(existing)
        embeddings = existing
        print(f"已加载 {start_idx} 个 embeddings，继续从第 {start_idx + 1} 条")
    else:
        embeddings = []
        start_idx = 0
        print(f"开始全新生成 embeddings")

    # 主循环，带进度条
    batch_start_time = time.time()  # 在循环开始前加这行

    for i in tqdm(range(start_idx, len(sequences)), desc="生成 ESM2 embeddings"):
        seq = sequences[i]
        emb = extract_protein_embedding(seq, model, tokenizer)
        embeddings.append(emb)

        # 每到一个保存点，保存并打印该批次时间
        if (i + 1) % save_every == 0 or (i + 1) == len(sequences):
            np.save(save_path, np.array(embeddings, dtype=object))
            batch_time = time.time() - batch_start_time
            print(f"已保存至 {save_path}，共 {i + 1} 个，本批耗时 {batch_time:.2f}s")
            batch_start_time = time.time()  # 重置时间戳

    return np.array(embeddings), labels


# 加载模型
esm_path = "d:/Users/Siqiniq/.cache/huggingface/hub/models--facebook--esm2_t33_650M_UR50D/snapshots/08e4846e537177426273712802403f7ba8261b6c/"
model, tokenizer = load_esm_components(esm_path)

# 读取数据
df = pd.read_csv("data/ToxinPred.csv")

# 执行增强版 embedding 提取
X, y = generate_embeddings_in_batches(
    df,
    model,
    tokenizer,
    save_path="data/esm2_toxin_embeddings.npy",  # 自定义保存路径
    save_every=50                                 # 每处理50条保存一次
)

