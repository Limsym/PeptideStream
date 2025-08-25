# filter_bad_samples.py
"""
预处理脚本：清理明显错误的突变样本
输入: data/s2648.csv
输出: data/s2648_clean.csv
"""
import pandas as pd
import os


def is_valid_mutation(mut: str) -> bool:
    if not mut or len(mut) < 3:
        return False
    import re
    return re.match(r"^[A-Z][0-9]+[A-Z]$", mut) is not None


def filter_bad_samples(input_csv: str = "../data/s2648.csv", output_csv: str = "../data/s2648_clean.csv"):
    print(f"📊 正在加载数据: {input_csv}")
    df = pd.read_csv(input_csv, sep=",")  # 修正分隔符为逗号
    before = len(df)
    print(f"原始数据: {before} 条记录")
    
    print("🧹 开始过滤无效突变...")
    df = df[df["MUT"].apply(is_valid_mutation)]
    after = len(df)
    
    print(f"✅ 过滤完成: {before} -> {after}, 删除 {before - after} 条无效记录")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"💾 已保存到: {output_csv}")


if __name__ == "__main__":
    filter_bad_samples()