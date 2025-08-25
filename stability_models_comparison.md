# Stability模型脚本对比分析

## 概述
PeptideStream项目中有两个稳定性预测模型脚本：`stability_model.py` 和 `stability_model_esm.py`。它们都用于训练蛋白质稳定性预测模型，但在实现方式上有重要区别。

## 脚本功能对比

### 1. **stability_model.py** - 基础版本
**功能**: 使用模拟embedding训练稳定性预测模型

#### 主要特点：
- **数据源**: `data/s2648.csv`
- **特征**: 使用随机生成的1280维向量模拟ESM2 embedding
- **模型**: GradientBoostingRegressor (梯度提升回归)
- **用途**: 原型验证和快速测试

#### 代码结构：
```python
# 1. 加载S2648数据集
df = pd.read_csv("data/s2648.csv", sep=",")
df = df[["PDB", "CHAIN", "MUT", "DDG"]].dropna()

# 2. 标签：DDG值（稳定性变化）
y = df["DDG"].values

# 3. 模拟embedding（占位符）
X = np.random.rand(len(df), 1280)  # 1280维随机向量

# 4. 训练GradientBoostingRegressor
model = GradientBoostingRegressor(n_estimators=200, max_depth=5)
```

### 2. **stability_model_esm.py** - 完整版本
**功能**: 使用真实ESM2 embedding训练稳定性预测模型

#### 主要特点：
- **数据源**: `data/s2648_clean.csv` (经过filter_bad_samples.py预处理)
- **特征**: 真实的ESM2蛋白质embedding
- **模型**: Ridge回归（线性模型）
- **用途**: 生产环境使用的完整实现

#### 代码结构：
```python
# 1. 加载数据并生成序列key
def load_data():
    for _, row in tqdm(df.iterrows()):
        key = f"{row['PDB']}_{row['CHAIN']}_{row['MUT']}"
        seqs.append(key)
        labels.append(row["DDG"])

# 2. 使用EmbeddingPipeline生成真实embedding
pipeline = EmbeddingPipeline(batch_size=4, device="cpu")
embeddings = pipeline.embed_sequences(seqs, resume=True)

# 3. 训练Ridge回归模型
model = Ridge(alpha=1.0)
```

## 关键差异对比

| 特性 | stability_model.py | stability_model_esm.py |
|------|-------------------|------------------------|
| **Embedding类型** | 随机生成（模拟） | 真实ESM2 embedding |
| **模型算法** | GradientBoostingRegressor | Ridge回归 |
| **数据处理** | 简单加载 | 批处理+断点续训 |
| **进度显示** | 无 | tqdm进度条 |
| **模型保存** | joblib | pickle |
| **代码结构** | 脚本式 | 函数式+模块化 |
| **生产就绪** | ❌ | ✅ |

## 联系和演进关系

### 1. **演进关系**
```
stability_model.py (原型) → stability_model_esm.py (生产版本)
```

- `stability_model.py` 是早期原型，用于验证模型训练流程
- `stability_model_esm.py` 是完整实现，用于实际生产环境

### 2. **共同目标**
- 都预测蛋白质突变对稳定性的影响（ΔΔG值）
- 都使用S2648数据集（蛋白质稳定性数据集）
- 都输出回归模型用于预测

### 3. **数据流程对比**

#### stability_model.py 流程：
```
data/s2648.csv → 提取DDG标签 → 生成随机embedding → 训练GBR → 保存模型
```

#### stability_model_esm.py 流程：
```
data/s2648.csv → filter_bad_samples.py → data/s2648_clean.csv → 生成ESM2 embedding → 训练Ridge → 保存模型
```

## 使用场景

### stability_model.py 适用场景：
- 🧪 **快速原型验证**
- 🚀 **快速测试模型训练流程**
- 📊 **验证数据处理逻辑**
- ⚡ **不需要真实embedding的测试**

### stability_model_esm.py 适用场景：
- 🏭 **生产环境部署**
- 🔬 **真实蛋白质稳定性预测**
- 📈 **需要准确预测结果**
- 💾 **大规模数据处理**

## 性能对比

### 训练时间：
- `stability_model.py`: 快速（无真实embedding生成）
- `stability_model_esm.py`: 较慢（需要生成ESM2 embedding）

### 预测准确性：
- `stability_model.py`: 低（使用随机特征）
- `stability_model_esm.py`: 高（使用真实蛋白质特征）

### 内存使用：
- `stability_model.py`: 低
- `stability_model_esm.py`: 中等（批处理优化）

## 建议使用策略

### 1. **数据预处理**
```python
# 首先运行数据清理脚本
python models/filter_bad_samples.py
```

### 2. **开发阶段**
```python
# 使用stability_model.py进行快速迭代
python models/stability_model.py
```

### 3. **生产阶段**
```python
# 使用stability_model_esm.py进行正式训练
python models/stability_model_esm.py
```

### 4. **预测使用**
```python
# 使用训练好的模型进行预测
from models.stability_predictor import StabilityPredictor
predictor = StabilityPredictor("models/stability_predictor.pkl")
result = predictor.predict(embedding)
```

## 总结

- **`stability_model.py`**: 快速原型，用于验证和测试
- **`stability_model_esm.py`**: 完整实现，用于生产环境
- **联系**: 后者是前者的演进版本，从模拟数据升级到真实embedding
- **建议**: 开发时用前者快速验证，部署时用后者获得准确结果
