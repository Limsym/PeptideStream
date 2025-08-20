# stability_model_esm.py
"""
纯序列版（MVP）——使用 UniProt / RCSB FASTA 作为序列来源
功能：
- 从 CSV 读取突变数据（需要列：PDB, CHAIN, MUT, DDG；若有 UNIPROT 列会优先使用）
- 拉取参考序列（优先 UniProt，再回退 RCSB FASTA），本地缓存
- 过滤：位置超长 / 残基不匹配（记录并跳过）
- 用 ESM 提取 WT / MUT embedding，特征为 emb_mut - emb_wt
- 训练简单的 GradientBoostingRegressor 并保存模型
注：不使用任何 PDB 解析或 3D 结构
"""
import os
import io
import time
import requests
import numpy as np
import pandas as pd
from typing import Optional, Tuple
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# 假定你已有的模块接口（不改动）
from embedding.esm_embedding_extractor import extract_protein_embedding, load_esm_components

# ---------- 配置 ----------
DATA_PATH = "../data/S2648.csv"
MODEL_OUT = "../models/stability_predictor.pkl"
ESM_CACHE_PATH = "d:/Users/Siqiniq/.cache/huggingface/hub/models--facebook--esm2_t33_650M_UR50D/snapshots/08e4846e537177426273712802403f7ba8261b6c/"
SEQ_CACHE_DIR = "./seq_cache"
MAX_SAMPLES = 100        # 上限以加速调试，改为 None 或较大值以处理全部
REQUEST_TIMEOUT = 15
# -------------------------

os.makedirs(SEQ_CACHE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)

def cache_path_for(identifier: str) -> str:
    safe = identifier.replace("/", "_").replace(":", "_")
    return os.path.join(SEQ_CACHE_DIR, f"{safe}.fasta")

def get_uniprot_sequence(uniprot_id: str) -> Optional[str]:
    """
    从 UniProt 拉取 fasta（REST）。返回序列字符串或 None。
    """
    if not uniprot_id:
        return None
    path = cache_path_for(uniprot_id)
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                lines = f.read().splitlines()
                seq = "".join(line.strip() for line in lines if not line.startswith(">"))
                return seq
        except Exception:
            pass
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    try:
        r = requests.get(url, timeout=REQUEST_TIMEOUT)
        if r.ok and r.text:
            fasta = r.text
            seq = "".join(line.strip() for line in fasta.splitlines() if not line.startswith(">"))
            with open(path, "w") as f:
                f.write(fasta)
            return seq
    except Exception:
        pass
    return None

def get_rcsb_fasta_sequence(pdb_id: str, chain: Optional[str] = None) -> Optional[str]:
    """
    拉取 RCSB 提供的 FASTA（可能包含多个 chain）。若指定 chain，尝试匹配；否则若仅一条序列则返回它。
    """
    if not pdb_id:
        return None
    key = f"rcsb_{pdb_id}_{chain or ''}"
    path = cache_path_for(key)
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                lines = f.read().splitlines()
                seq = "".join(line.strip() for line in lines if not line.startswith(">"))
                return seq
        except Exception:
            pass
    url = f"https://www.rcsb.org/fasta/entry/{pdb_id}"
    try:
        r = requests.get(url, timeout=REQUEST_TIMEOUT)
        if r.ok and r.text:
            fasta_text = r.text
            # 解析多个序列（按 header 中的 chain 标识做简单匹配）
            seqs = {}
            cur_header = None
            cur_seq = []
            for line in fasta_text.splitlines():
                if line.startswith(">"):
                    if cur_header is not None:
                        seqs[cur_header] = "".join(cur_seq)
                    header = line[1:].strip()
                    token = header.split()[0]
                    ch = None
                    if "_" in token:
                        parts = token.split("_"); ch = parts[1]
                    elif ":" in token:
                        parts = token.split(":"); ch = parts[1]
                    elif len(token) > 4 and token[4].isalpha():
                        ch = token[4]
                    else:
                        toks = header.split()
                        ch = toks[1] if len(toks) > 1 else None
                    cur_header = ch or token
                    cur_seq = []
                else:
                    cur_seq.append(line.strip())
            if cur_header is not None:
                seqs[cur_header] = "".join(cur_seq)
            # 缓存原始 FASTA
            with open(path, "w") as f:
                f.write(fasta_text)
            if chain and chain in seqs:
                return seqs[chain]
            if chain:
                # 忽略大小写
                for k, v in seqs.items():
                    if k and k.lower() == chain.lower():
                        return v
            # 没有指定 chain，且仅一条序列则返回
            if len(seqs) == 1:
                return next(iter(seqs.values()))
    except Exception:
        pass
    return None

def parse_mutation(mut: str) -> Optional[Tuple[str, int, str]]:
    """
    解析突变字符串（只处理单个突变），例如 "C30S" -> ("C", 29, "S") (0-based pos).
    若格式异常返回 None。
    """
    if not isinstance(mut, str):
        return None
    mut = mut.strip()
    # 允许像 "C30S" 或 "C30S;other" 的情况 — 只取第一项
    mut_main = mut.split(";")[0].strip()
    if len(mut_main) < 3:
        return None
    orig = mut_main[0]
    new = mut_main[-1]
    try:
        pos = int(mut_main[1:-1]) - 1
        return orig, pos, new
    except Exception:
        return None

def apply_mutation_on_sequence(seq: str, orig: str, pos: int, new: str) -> Optional[str]:
    if seq is None:
        return None
    if pos < 0 or pos >= len(seq):
        return None
    if seq[pos] != orig:
        return None
    return seq[:pos] + new + seq[pos+1:]

def get_reference_sequence(row: pd.Series) -> Tuple[Optional[str], str]:
    """
    根据行优先策略获取参考序列：
    - 若有 UNIPROT 列 -> 使用 UniProt
    - 否则尝试把 PDB 当作 UniProt id 去拉取（轻量做法）
    - 再否则回退到 RCSB FASTA（使用 PDB id + CHAIN）
    返回 (sequence or None, source_description)
    """
    # 优先 UNIPROT 列
    if "UNIPROT" in row and pd.notna(row["UNIPROT"]):
        seq = get_uniprot_sequence(str(row["UNIPROT"]).strip())
        if seq:
            return seq, f"uniprot:{row['UNIPROT']}"
    # 尝试把 PDB 字段当作 UniProt id（在某些数据来源 PDB 列实际保存的是 UniProt）
    if "PDB" in row and pd.notna(row["PDB"]):
        candidate = str(row["PDB"]).strip()
        seq = get_uniprot_sequence(candidate)
        if seq:
            return seq, f"uniprot_from_pdbcol:{candidate}"
        # 回退到 rcsb fasta（需 chain 信息）
        chain = str(row["CHAIN"]).strip() if "CHAIN" in row and pd.notna(row["CHAIN"]) else None
        seq = get_rcsb_fasta_sequence(candidate, chain)
        if seq:
            return seq, f"rcsb:{candidate}:{chain}"
    return None, "none"

def main():
    # 1. 读取数据
    df = pd.read_csv(DATA_PATH)
    # 保留必须列
    expected_cols = {"PDB", "CHAIN", "MUT", "DDG"}
    if not expected_cols.issubset(set(df.columns)):
        raise ValueError(f"CSV 必须包含列：{expected_cols}，当前文件列为 {set(df.columns)}")
    df = df[["PDB", "CHAIN", "MUT", "DDG"]].dropna().reset_index(drop=True)

    # 加载 ESM（假设该函数存在并返回 model, tokenizer）
    print("加载 ESM 组件 ...")
    esm_model, esm_tokenizer = load_esm_components(ESM_CACHE_PATH)
    print("ESM 加载完成。")

    X_list = []
    y_list = []
    stats = {
        "total_rows": 0,
        "no_seq": 0,
        "bad_mut_format": 0,
        "pos_out_of_range": 0,
        "residue_mismatch": 0,
        "success": 0
    }

    # 遍历处理
    for idx, row in df.iterrows():
        if MAX_SAMPLES and stats["success"] >= MAX_SAMPLES:
            break
        stats["total_rows"] += 1

        seq, src = get_reference_sequence(row)
        if seq is None:
            stats["no_seq"] += 1
            # 记录并跳过
            print(f"跳过行 {idx}: 找不到参考序列（PDB={row['PDB']}, CHAIN={row['CHAIN']}）")
            continue

        parsed = parse_mutation(row["MUT"])
        if parsed is None:
            stats["bad_mut_format"] += 1
            print(f"跳过行 {idx}: 突变格式无法解析：{row['MUT']}")
            continue
        orig, pos, new = parsed

        # 检查位置范围
        if pos < 0 or pos >= len(seq):
            stats["pos_out_of_range"] += 1
            print(f"⚠️ {row['PDB']} {row['CHAIN']} 突变位置超出序列长度，跳过：{row['MUT']} (len={len(seq)})")
            continue

        # 检查残基是否匹配
        if seq[pos] != orig:
            stats["residue_mismatch"] += 1
            print(f"⚠️ {row['PDB']} {row['CHAIN']} 序列残基不匹配，跳过：{row['MUT']} (expected {orig}, actual {seq[pos]})")
            continue

        # 构造突变后序列（WT 和 MUT）
        wt_seq = seq
        mut_seq = apply_mutation_on_sequence(seq, orig, pos, new)
        if mut_seq is None:
            print(f"⚠️ 无法在参考序列上施加突变，跳过行 {idx}: {row['MUT']}")
            continue

        # 提取 embedding（稳健调用）
        try:
            emb_wt = extract_protein_embedding(wt_seq, esm_model, esm_tokenizer)
            emb_mut = extract_protein_embedding(mut_seq, esm_model, esm_tokenizer)
        except Exception as e:
            print(f"⚠️ ESM embedding 失败（行 {idx}）：{e}")
            continue

        # 检查 emb 形状一致
        if emb_wt is None or emb_mut is None:
            print(f"⚠️ ESM 返回空 embedding，跳过（行 {idx}）")
            continue
        if emb_wt.shape != emb_mut.shape:
            print(f"⚠️ ESM embedding 尺寸不一致，跳过（行 {idx}）")
            continue

        diff = emb_mut - emb_wt
        # 确保是 1D 数组
        diff = np.asarray(diff).reshape(-1)
        X_list.append(diff.astype(np.float32))
        y_list.append(float(row["DDG"]))
        stats["success"] += 1

        if stats["success"] % 20 == 0:
            print(f"已处理成功样本数：{stats['success']} (迭代至行 {idx})")

    # 完成：统计
    print("处理结束，统计：")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    if stats["success"] == 0:
        raise RuntimeError("没有成功的样本，无法训练模型。请检查数据源和突变格式。")

    # 把 X_list 转为矩阵
    try:
        X = np.vstack(X_list)   # shape (n_samples, emb_dim)
    except Exception as e:
        # 若发生堆叠问题，尝试逐项 pad（不推荐）
        raise RuntimeError(f"无法将 embedding 列表堆叠成矩阵：{e}")

    y = np.array(y_list, dtype=np.float32)

    # 划分训练/验证集并训练回归模型
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"数据准备完毕：训练样本 {len(X_train)}，验证样本 {len(X_val)}。开始训练回归模型...")

    model_reg = GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42)
    model_reg.fit(X_train, y_train)

    y_pred = model_reg.predict(X_val)
    print(f"✅ MSE: {mean_squared_error(y_val, y_pred):.4f} | R²: {r2_score(y_val, y_pred):.4f}")

    joblib.dump(model_reg, MODEL_OUT)
    print(f"✅ 模型已保存为 '{MODEL_OUT}'")

if __name__ == "__main__":
    main()
