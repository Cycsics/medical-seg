import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalMemoryModule(nn.Module):
    """时序记忆模块，用于增强内窥镜图像中伤口边缘的时序信息获取"""
    
    def __init__(self, in_channels, memory_channels=256, memory_size=5, edge_kernel_size=3):
        """
        初始化时序记忆模块
        
        Args:
            in_channels (int): 输入通道数
            memory_channels (int): 记忆通道数
            memory_size (int): 记忆帧数
            edge_kernel_size (int): 边缘检测卷积核大小
        """
        super().__init__()
        self.memory_size = memory_size
        self.memory_channels = memory_channels
        
        # 边缘特征提取
        self.edge_conv = nn.Sequential(
            nn.Conv2d(in_channels, memory_channels, edge_kernel_size, padding=edge_kernel_size//2),
            nn.BatchNorm2d(memory_channels),
            nn.ReLU(inplace=True)
        )
        
        # 时序记忆更新
        self.memory_update = nn.GRUCell(
            memory_channels * 2,  # 当前特征 + 记忆特征
            memory_channels
        )
        
        # 记忆注意力
        self.memory_attention = nn.Sequential(
            nn.Conv2d(memory_channels * 2, memory_channels // 8, 1),
            nn.BatchNorm2d(memory_channels // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(memory_channels // 8, memory_channels, 1),
            nn.Sigmoid()
        )
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(memory_channels * 2, memory_channels, 1),
            nn.BatchNorm2d(memory_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(memory_channels, in_channels, 1)
        )
        
        # 初始化记忆
        self.register_buffer('memory', None)
        
    def init_memory(self, batch_size, height, width, device):
        """初始化记忆"""
        self.memory = torch.zeros(
            batch_size, self.memory_channels, height, width,
            device=device
        )
        
    def update_memory(self, current_feat, memory):
        """更新记忆"""
        # 计算注意力权重
        attn = self.memory_attention(torch.cat([current_feat, memory], dim=1))
        
        # 更新记忆
        memory = memory * (1 - attn) + current_feat * attn
        return memory
        
    def forward(self, x, reset_memory=False):
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入特征图 [B, C, H, W]
            reset_memory (bool): 是否重置记忆
            
        Returns:
            torch.Tensor: 增强后的特征图
        """
        # 训练时直接返回边缘特征
        if self.training:
            edge_feat = self.edge_conv(x)
            return edge_feat + x  # 残差连接
            
        # 推理时使用记忆机制
        batch_size, _, height, width = x.shape
        device = x.device
        
        # 初始化或重置记忆
        if self.memory is None or reset_memory:
            self.init_memory(batch_size, height, width, device)
            
        # 提取边缘特征
        edge_feat = self.edge_conv(x)
        
        # 更新记忆
        self.memory = self.update_memory(edge_feat, self.memory)
        
        # 特征融合
        out = self.fusion(torch.cat([edge_feat, self.memory], dim=1))
        
        return out + x  # 残差连接
        
class TemporalMemoryBlock(nn.Module):
    """时序记忆模块，用于增强时序特征提取"""
    
    def __init__(self, in_channels, memory_channels, num_modules=3):
        """
        初始化时序记忆模块
        
        Args:
            in_channels (int): 输入通道数
            memory_channels (int): 记忆通道数
            num_modules (int): 模块数量
        """
        super().__init__()
        self.modules = nn.ModuleList([
            TemporalMemoryModule(in_channels, memory_channels)
            for _ in range(num_modules)
        ])
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入特征图 [B, C, H, W]
            
        Returns:
            torch.Tensor: 增强后的特征图
        """
        for module in self.modules:
            x = module(x)
        return x 