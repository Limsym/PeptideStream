# to do

---

## [x] 训练进度条显示
pip install tqdm

Epoch 1/30: 100%|██████████| 125/125 [00:10<00:00, 12.34it/s, loss=0.432]


## [x] 过拟合问题。
解决方案：实现 Early Stopping
热插拔训练 检查checkpoint，导入best_val_loss，将其作为参数传入train()