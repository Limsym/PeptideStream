# to do

---

## [x] 训练进度条显示
pip install tqdm

Epoch 1/30: 100%|██████████| 125/125 [00:10<00:00, 12.34it/s, loss=0.432]


## [x] 过拟合问题。
解决方案：实现 Early Stopping
热插拔训练 检查checkpoint，导入best_val_loss，将其作为参数传入train()

[x] 下载 ESM 模型
安装 huggingface_hub
pip install -U huggingface_hub

--从 cli 转向 hf 命令--
| ⚠️  Warning: 'huggingface-cli download' is deprecated. Use 'hf download' instead.
 
[x] 调用 ESM 并保存 embeddings

[] prepare the stability module
- dataset: FireProtDB
- feature
- train
- immobilize
- export