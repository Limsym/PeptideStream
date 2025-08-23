# stability_model_esm.py

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
from embedding.esm_embedding_extractor import extract_protein_embedding, load_esm_components
import re

# ---------- 配置 ----------
DATA_PATH = "../data/S2648.csv"
MODEL_OUT = "../models/stability_predictor.pkl"
ESM_CACHE_PATH = "d:/Users/Siqiniq/.cache/huggingface/hub/models--facebook--esm2_t33_650M_UR50D/snapshots/08e4846e537177426273712802403f7ba8261b6c/"
SEQ_CACHE_DIR = "./seq_cache"
MAX_SAMPLES = None
REQUEST_TIMEOUT = 15

# 新增参数：错误输出阈值
MAX_ERROR_LOG = 10

os.makedirs(SEQ_CACHE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)

# --- 突变解析 ---
_mut_re = re.compile(r'^([A-Z])(\d+)([A-Z])$')
def parse_mutation(mut: str) -> Optional[Tuple[str, int, str]]:
    if not isinstance(mut, str):
        return None
    m = _mut_re.match(mut.strip())
    if not m:
        return None
    wt, pos, mut_new = m.groups()
    return wt, int(pos) - 1, mut_new

# --- 应用突变 ---
def apply_mutation_on_sequence(seq: str, orig: str, pos: int, new: str) -> Optional[str]:
    if seq is None or pos < 0 or pos >= len(seq):
        return None
    if seq[pos] != orig:
        return None
    return seq[:pos] + new + seq[pos+1:]

# --- 序列获取（简化：仅保留 UniProt / RCSB FASTA） ---
def cache_path_for(identifier: str) -> str:
    safe = identifier.replace("/", "_").replace(":", "_")
    return os.path.join(SEQ_CACHE_DIR, f"{safe}.fasta")

def get_uniprot_sequence(uniprot_id: str) -> Optional[str]:
    if not uniprot_id:
        return None
    path = cache_path_for(uniprot_id)
    if os.path.exists(path):
        with open(path, "r") as f:
            lines = f.read().splitlines()
            return "".join(line.strip() for line in lines if not line.startswith(">"))
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    try:
        r = requests.get(url, timeout=REQUEST_TIMEOUT)
        if r.ok:
            fasta = r.text
            seq = "".join(line.strip() for line in fasta.splitlines() if not line.startswith(">"))
            with open(path, "w") as f:
                f.write(fasta)
            return seq
    except Exception:
        return None
    return None

def get_rcsb_fasta_sequence(pdb_id: str, chain: Optional[str] = None) -> Optional[str]:
    if not pdb_id:
        return None
    key = f"rcsb_{pdb_id}_{chain or ''}"
    path = cache_path_for(key)
    if os.path.exists(path):
        with open(path, "r") as f:
            lines = f.read().splitlines()
            return "".join(line.strip() for line in lines if not line.startswith(">"))
    url = f"https://www.rcsb.org/fasta/entry/{pdb_id}"
    try:
        r = requests.get(url, timeout=REQUEST_TIMEOUT)
        if r.ok:
            fasta = r.text
            seqs = {}
            cur_header, cur_seq = None, []
            for line in fasta.splitlines():
                if line.startswith(">"):
                    if cur_header:
                        seqs[cur_header] = "".join(cur_seq)
                    token = line[1:].split()[0]
                    ch = token.split("_")[-1] if "_" in token else None
                    cur_header, cur_seq = ch or token, []
                else:
                    cur_seq.append(line.strip())
            if cur_header:
                seqs[cur_header] = "".join(cur_seq)
            with open(path, "w") as f:
                f.write(fasta)
            if chain and chain in seqs:
                return seqs[chain]
            if len(seqs) == 1:
                return next(iter(seqs.values()))
    except Exception:
        return None
    return None

def get_reference_sequence(row: pd.Series) -> Optional[str]:
    if "UNIPROT" in row and pd.notna(row["UNIPROT"]):
        seq = get_uniprot_sequence(str(row["UNIPROT"]))
        if seq:
            return seq
    if "PDB" in row and pd.notna(row["PDB"]):
        seq = get_rcsb_fasta_sequence(str(row["PDB"]), str(row["CHAIN"]))
        if seq:
            return seq
    return None

# --- 主流程 ---
def main():
    df = pd.read_csv(DATA_PATH)
    required = {"PDB", "CHAIN", "MUT", "DDG"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV 必须包含列 {required}")

    df = df[list(required)].dropna().reset_index(drop=True)

    print("加载 ESM 组件 ...")
    esm_model, esm_tokenizer = load_esm_components(ESM_CACHE_PATH)
    print("ESM 加载完成。")

    X_list, y_list = [], []
    stats = {"total": 0, "bad": 0, "success": 0}
    error_logs = []

    for idx, row in df.iterrows():
        stats["total"] += 1
        seq = get_reference_sequence(row)
        if not seq:
            stats["bad"] += 1
            if len(error_logs) < MAX_ERROR_LOG:
                error_logs.append(f"行{idx}: 无参考序列 {row['PDB']} {row['CHAIN']}")
            continue

        parsed = parse_mutation(row["MUT"])
        if not parsed:
            stats["bad"] += 1
            if len(error_logs) < MAX_ERROR_LOG:
                error_logs.append(f"行{idx}: 突变格式错误 {row['MUT']}")
            continue

        orig, pos, new = parsed
        if pos >= len(seq) or seq[pos] != orig:
            stats["bad"] += 1
            if len(error_logs) < MAX_ERROR_LOG:
                error_logs.append(f"行{idx}: 残基不符 {row['MUT']} (len={len(seq)})")
            continue

        mut_seq = apply_mutation_on_sequence(seq, orig, pos, new)
        if not mut_seq:
            stats["bad"] += 1
            continue

        emb_wt = extract_protein_embedding(seq, esm_model, esm_tokenizer)
        emb_mut = extract_protein_embedding(mut_seq, esm_model, esm_tokenizer)
        if emb_wt is None or emb_mut is None:
            stats["bad"] += 1
            continue

        diff = (emb_mut - emb_wt).reshape(-1)
        X_list.append(diff.astype(np.float32))
        y_list.append(float(row["DDG"]))
        stats["success"] += 1

        if stats["success"] % 20 == 0:
            print(f"已处理成功样本数：{stats['success']}")

    # 打印少量错误日志
    if error_logs:
        print("⚠️ 样本错误示例（最多显示前10条）：")
        for log in error_logs:
            print("  ", log)

    print("处理结束，统计：", stats)

    if not X_list:
        raise RuntimeError("无有效样本")

    X = np.vstack(X_list)
    y = np.array(y_list)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"训练 {len(X_train)}，验证 {len(X_val)}")

    model = GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    print(f"✅ MSE {mean_squared_error(y_val, y_pred):.4f} | R² {r2_score(y_val, y_pred):.4f}")
    joblib.dump(model, MODEL_OUT)
    print(f"✅ 模型已保存 {MODEL_OUT}")

if __name__ == "__main__":
    main()
