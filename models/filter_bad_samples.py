# filter_bad_samples_with_logs.py
"""
预处理脚本：清理明显错误的突变样本，并记录被删除的记录及其原因。
输入: ../data/s2648.csv
输出: ../data/s2648_clean.csv (有效记录)
      ../data/s2648_bad.csv   (无效记录)
"""
import pandas as pd
import os
import re


def is_valid_mutation_format(mut: str) -> bool:
    """验证突变字符串的格式，例如 'C30S'。"""
    if not mut or len(mut) < 3:
        return False
    return re.match(r"^[A-Z][0-9]+[A-Z]$", mut) is not None


def get_amino_acid_from_fasta(pdb_id: str, chain: str, position: int, seq_cache_path: str) -> str:
    """从FASTA文件中获取指定位置的氨基酸。"""
    fasta_file = os.path.join(seq_cache_path, f"rcsb_{pdb_id}_{chain}.fasta")
    if not os.path.exists(fasta_file):
        return None

    try:
        with open(fasta_file, 'r') as f:
            lines = f.readlines()
            sequence = "".join(line.strip() for line in lines if not line.startswith('>'))

            if 0 < position <= len(sequence):
                return sequence[position - 1]
            else:
                return None
    except Exception:
        return None


def filter_bad_samples(input_csv: str = "../data/s2648.csv", output_csv_clean: str = "../data/s2648_clean.csv",
                       output_csv_bad: str = "../data/s2648_bad.csv"):
    """主过滤函数，包含格式和FASTA验证，并分离有效和无效记录。"""
    seq_cache_path = os.path.join("..", "models", "seq_cache")

    print(f"📊 正在加载数据: {input_csv}")
    df = pd.read_csv(input_csv, sep=",")
    before = len(df)
    print(f"   原始数据: {before} 条记录")

    print("🧹 开始过滤无效突变...")

    # 添加新列来存储验证结果和FASTA残基
    df['is_valid'] = False
    df['Correct_FASTA_Residue'] = None

    for index, row in df.iterrows():
        mut_str = row["MUT"]
        pdb_id = row["PDB"]
        chain = row["CHAIN"]

        # 验证格式
        if not is_valid_mutation_format(mut_str):
            df.at[index, 'is_valid'] = False
            continue

        # 解析突变字符串，提取原始氨基酸和位置
        original_aa = mut_str[0]
        try:
            position = int(re.search(r"\d+", mut_str).group(0))
        except (ValueError, AttributeError):
            df.at[index, 'is_valid'] = False
            continue

        # 从FASTA文件中获取指定位置的氨基酸
        fasta_aa = get_amino_acid_from_fasta(pdb_id, chain, position, seq_cache_path)

        # 存储FASTA中的残基
        df.at[index, 'Correct_FASTA_Residue'] = fasta_aa

        # 验证 FASTA 文件中的氨基酸是否与突变前的氨基酸一致
        if fasta_aa is not None and fasta_aa == original_aa:
            df.at[index, 'is_valid'] = True
        else:
            df.at[index, 'is_valid'] = False
            print(f"警告：删除记录 - PDB: {pdb_id}, MUT: {mut_str}。原因：FASTA验证失败。FASTA中该位置残基为：{fasta_aa}")

    # 将DataFrame拆分为有效和无效记录
    final_df = df[df['is_valid']].drop(columns=['is_valid', 'Correct_FASTA_Residue'])
    bad_df = df[~df['is_valid']].drop(columns=['is_valid'])

    after = len(final_df)

    print(f"✅ 过滤完成: {before} -> {after}, 删除 {before - after} 条无效记录")

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_csv_clean), exist_ok=True)

    # 保存有效记录
    final_df.to_csv(output_csv_clean, index=False)
    print(f"💾 有效记录已保存到: {output_csv_clean}")

    # 保存被删除的记录
    if not bad_df.empty:
        bad_df.to_csv(output_csv_bad, index=False)
        print(f"🗑️ 无效记录已保存到: {output_csv_bad}")


if __name__ == "__main__":
    filter_bad_samples()