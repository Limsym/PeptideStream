import os
from huggingface_hub import snapshot_download, hf_hub_download
from transformers import EsmModel, EsmTokenizer

def load_esm_model(
    model_name: str = "facebook/esm2_t33_650M_UR50D",
    cache_dir: str = "./models/esm2_t33_650M_UR50D",
    local_only: bool = False
):
    print(f"🧠 准备加载 ESM 模型：{model_name}")
    print("📦 模型大小约为 2.5GB，请确保网络良好并有足够磁盘空间。\n")

    # 如果开启本地加载，则只检查缓存是否存在
    if local_only:
        if not os.path.exists(cache_dir):
            raise FileNotFoundError(f"⚠️ 指定的缓存路径 {cache_dir} 不存在。请先联网运行一次下载。")
        print(f"✅ 从本地路径加载模型：{cache_dir}")
    else:
        print(f"⬇️ 正在检查或下载模型到本地缓存目录：{cache_dir}")
        snapshot_download(repo_id=model_name, local_dir=cache_dir, local_dir_use_symlinks=False)
        print(f"✅ 模型下载完成或已存在。路径：{cache_dir}")

    # 加载模型和tokenizer
    tokenizer = EsmTokenizer.from_pretrained(cache_dir)
    model = EsmModel.from_pretrained(cache_dir)

    print("🚀 模型和Tokenizer已加载完毕，可用于推理。")
    return tokenizer, model
