# stability_predictor_pytorch.py
"""
PyTorch版本的稳定性预测器
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Union, List


class StabilityMLP(nn.Module):
    """稳定性预测MLP模型"""
    
    def __init__(self, input_dim=1280, hidden_dims=[512, 256, 128], dropout=0.2):
        super(StabilityMLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # 构建隐藏层
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze(-1)


class StabilityPredictorPyTorch:
    """PyTorch版本的稳定性预测器"""
    
    def __init__(self, model_path: str = "models/stability_predictor_pytorch.pth", 
                 device: str = "auto"):
        """
        初始化预测器
        
        Args:
            model_path: 模型文件路径
            device: 设备选择 ("auto", "cpu", "cuda")
        """
        self.device = self._get_device(device)
        self.model = self._load_model(model_path)
        self.model.eval()
        
    def _get_device(self, device: str) -> torch.device:
        """获取设备"""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            return torch.device(device)
    
    def _load_model(self, model_path: str) -> StabilityMLP:
        """加载模型"""
        try:
            # 创建模型实例
            model = StabilityMLP()
            
            # 加载模型权重
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✅ 模型加载成功，最佳验证损失: {checkpoint.get('best_val_loss', 'N/A')}")
            else:
                model.load_state_dict(checkpoint)
                print("✅ 模型加载成功")
                
            return model
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise
    
    def predict(self, embedding: Union[np.ndarray, torch.Tensor]) -> float:
        """
        预测单个embedding的稳定性
        
        Args:
            embedding: ESM2 embedding (numpy array or tensor)
            
        Returns:
            float: 预测的ΔΔG值
        """
        # 转换为tensor
        if isinstance(embedding, np.ndarray):
            embedding = torch.FloatTensor(embedding)
        
        # 确保维度正确
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)  # 添加batch维度
        
        # 移动到设备
        embedding = embedding.to(self.device)
        
        # 预测
        with torch.no_grad():
            prediction = self.model(embedding)
            
        return float(prediction.cpu().numpy())
    
    def predict_batch(self, embeddings: Union[np.ndarray, torch.Tensor]) -> List[float]:
        """
        批量预测多个embedding的稳定性
        
        Args:
            embeddings: ESM2 embeddings数组
            
        Returns:
            List[float]: 预测的ΔΔG值列表
        """
        # 转换为tensor
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.FloatTensor(embeddings)
        
        # 移动到设备
        embeddings = embeddings.to(self.device)
        
        # 批量预测
        with torch.no_grad():
            predictions = self.model(embeddings)
            
        return predictions.cpu().numpy().tolist()
    
    def predict_with_confidence(self, embedding: Union[np.ndarray, torch.Tensor], 
                              n_samples: int = 10) -> tuple:
        """
        使用Monte Carlo Dropout进行不确定性估计
        
        Args:
            embedding: ESM2 embedding
            n_samples: Monte Carlo采样次数
            
        Returns:
            tuple: (预测值, 不确定性)
        """
        # 转换为tensor
        if isinstance(embedding, np.ndarray):
            embedding = torch.FloatTensor(embedding)
        
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        
        embedding = embedding.to(self.device)
        
        # 启用dropout进行Monte Carlo采样
        self.model.train()
        predictions = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.model(embedding)
                predictions.append(pred.cpu().numpy())
        
        # 恢复评估模式
        self.model.eval()
        
        predictions = np.array(predictions).flatten()
        mean_pred = float(np.mean(predictions))
        uncertainty = float(np.std(predictions))
        
        return mean_pred, uncertainty


# 使用示例
if __name__ == "__main__":
    # 创建预测器
    predictor = StabilityPredictorPyTorch()
    
    # 模拟一个embedding
    dummy_embedding = np.random.randn(1280)
    
    # 单个预测
    result = predictor.predict(dummy_embedding)
    print(f"预测结果: {result:.4f}")
    
    # 批量预测
    dummy_embeddings = np.random.randn(5, 1280)
    results = predictor.predict_batch(dummy_embeddings)
    print(f"批量预测结果: {[f'{r:.4f}' for r in results]}")
    
    # 带不确定性的预测
    pred, uncertainty = predictor.predict_with_confidence(dummy_embedding)
    print(f"预测: {pred:.4f} ± {uncertainty:.4f}")
