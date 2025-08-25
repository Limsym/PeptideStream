# stability_predictor_pytorch_enhanced.py
"""
增强版PyTorch稳定性预测器
支持集成模型预测
"""
import torch
import torch.nn as nn
import numpy as np
import os
from typing import Union, List


class HuberLoss(nn.Module):
    """Huber损失函数，对异常值更鲁棒"""
    
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta
    
    def forward(self, y_pred, y_true):
        error = y_pred - y_true
        abs_error = torch.abs(error)
        quadratic = torch.clamp(abs_error, max=self.delta)
        linear = abs_error - quadratic
        return torch.mean(0.5 * quadratic**2 + self.delta * linear)


class StabilityMLPEnhanced(nn.Module):
    """增强版稳定性预测MLP模型"""
    
    def __init__(self, input_dim=1280, hidden_dims=[512, 256, 128, 64], dropout=0.3):
        super(StabilityMLPEnhanced, self).__init__()
        
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


class StabilityPredictorPyTorchEnhanced:
    """增强版PyTorch稳定性预测器"""
    
    def __init__(self, model_path: str = "models/stability_predictor_pytorch_enhanced.pth", 
                 device: str = "auto"):
        """
        初始化增强版预测器
        
        Args:
            model_path: 集成模型信息文件路径
            device: 设备选择 ("auto", "cpu", "cuda")
        """
        self.device = self._get_device(device)
        self.ensemble_info = self._load_ensemble_info(model_path)
        self.models = self._load_ensemble_models()
        self.scaler = self.ensemble_info['scaler']
        
        print(f"✅ 增强版预测器加载成功")
        print(f"   📊 集成模型数量: {self.ensemble_info['n_models']}")
        print(f"   📈 训练性能: R² = {self.ensemble_info['ensemble_r2']:.4f}")
        print(f"   🚀 性能提升: +{self.ensemble_info['performance_improvement']:.4f}")
        
    def _get_device(self, device: str) -> torch.device:
        """获取设备"""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            return torch.device(device)
    
    def _load_ensemble_info(self, model_path: str) -> dict:
        """加载集成模型信息"""
        try:
            ensemble_info = torch.load(model_path, map_location=self.device)
            return ensemble_info
        except Exception as e:
            print(f"❌ 集成模型信息加载失败: {e}")
            raise
    
    def _load_ensemble_models(self) -> List[StabilityMLPEnhanced]:
        """加载所有集成模型"""
        models = []
        model_paths = self.ensemble_info['model_paths']
        
        for i, model_path in enumerate(model_paths):
            try:
                # 创建模型实例
                model = StabilityMLPEnhanced(
                    input_dim=1280,
                    hidden_dims=[512, 256, 128, 64],
                    dropout=0.3
                )
                
                # 加载模型权重
                checkpoint = torch.load(model_path, map_location=self.device)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.to(self.device)
                model.eval()
                
                models.append(model)
                print(f"✅ 模型 {i+1} 加载成功")
                
            except Exception as e:
                print(f"❌ 模型 {i+1} 加载失败: {e}")
                raise
        
        return models
    
    def predict(self, embedding: Union[np.ndarray, torch.Tensor]) -> float:
        """
        预测单个embedding的稳定性
        
        Args:
            embedding: 1280维ESM2 embedding
            
        Returns:
            预测的稳定性值 (ΔΔG)
        """
        # 数据预处理
        if isinstance(embedding, np.ndarray):
            embedding = torch.FloatTensor(embedding)
        
        # 标准化
        embedding_scaled = self.scaler.transform(embedding.unsqueeze(0).cpu().numpy())
        embedding_scaled = torch.FloatTensor(embedding_scaled).to(self.device)
        
        # 集成预测
        predictions = []
        with torch.no_grad():
            for model in self.models:
                pred = model(embedding_scaled)
                predictions.append(pred.cpu().numpy())
        
        # 返回平均预测
        return float(np.mean(predictions))
    
    def predict_batch(self, embeddings: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        批量预测多个embedding的稳定性
        
        Args:
            embeddings: 形状为 (n_samples, 1280) 的embedding数组
            
        Returns:
            预测的稳定性值数组
        """
        # 数据预处理
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.FloatTensor(embeddings)
        
        # 标准化
        embeddings_scaled = self.scaler.transform(embeddings.cpu().numpy())
        embeddings_scaled = torch.FloatTensor(embeddings_scaled).to(self.device)
        
        # 集成预测
        predictions = []
        with torch.no_grad():
            for model in self.models:
                pred = model(embeddings_scaled)
                predictions.append(pred.cpu().numpy())
        
        # 返回平均预测
        return np.mean(predictions, axis=0)
    
    def predict_with_confidence(self, embedding: Union[np.ndarray, torch.Tensor], 
                              n_samples: int = 10) -> tuple:
        """
        预测稳定性并估计不确定性（使用Monte Carlo Dropout）
        
        Args:
            embedding: 1280维ESM2 embedding
            n_samples: Monte Carlo采样次数
            
        Returns:
            (预测值, 不确定性估计)
        """
        # 数据预处理
        if isinstance(embedding, np.ndarray):
            embedding = torch.FloatTensor(embedding)
        
        # 标准化
        embedding_scaled = self.scaler.transform(embedding.unsqueeze(0).cpu().numpy())
        embedding_scaled = torch.FloatTensor(embedding_scaled).to(self.device)
        
        # Monte Carlo Dropout预测
        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                sample_predictions = []
                for model in self.models:
                    # 启用dropout进行不确定性估计
                    model.train()
                    pred = model(embedding_scaled)
                    sample_predictions.append(pred.cpu().numpy())
                    model.eval()
                
                # 集成预测
                ensemble_pred = np.mean(sample_predictions)
                predictions.append(ensemble_pred)
        
        # 计算预测值和不确定性
        mean_pred = np.mean(predictions)
        uncertainty = np.std(predictions)
        
        return float(mean_pred), float(uncertainty)
    
    def get_model_info(self) -> dict:
        """获取模型信息"""
        return {
            'n_models': self.ensemble_info['n_models'],
            'ensemble_r2': self.ensemble_info['ensemble_r2'],
            'ridge_r2': self.ensemble_info['ridge_r2'],
            'performance_improvement': self.ensemble_info['performance_improvement'],
            'device': str(self.device)
        }


if __name__ == "__main__":
    # 测试增强版预测器
    print("🧪 测试增强版稳定性预测器...")
    
    try:
        # 初始化预测器
        predictor = StabilityPredictorPyTorchEnhanced()
        
        # 创建测试embedding
        dummy_embedding = np.random.randn(1280)
        
        # 测试单次预测
        pred = predictor.predict(dummy_embedding)
        print(f"📊 单次预测: {pred:.4f}")
        
        # 测试批量预测
        dummy_embeddings = np.random.randn(5, 1280)
        batch_preds = predictor.predict_batch(dummy_embeddings)
        print(f"📊 批量预测: {batch_preds}")
        
        # 测试不确定性预测
        pred, uncertainty = predictor.predict_with_confidence(dummy_embedding)
        print(f"📊 不确定性预测: {pred:.4f} ± {uncertainty:.4f}")
        
        # 显示模型信息
        info = predictor.get_model_info()
        print(f"📋 模型信息: {info}")
        
        print("✅ 增强版预测器测试完成!")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        print("💡 请确保已运行 stability_model_pytorch_enhanced.py 训练集成模型")

