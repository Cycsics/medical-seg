import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import Segment
from .convlstm import ConvLSTM

class SegmentHeadLSTM(Segment):
    """YOLO11 LSTM segmentation head."""
    
    def __init__(self, nc=80, nm=32, npr=256, ch=(), lstm_kernel=3, seq_len=1, 
                 use_attention=True, use_edge_enhancement=True):
        super().__init__(nc, nm, npr, ch)
        self.seq_len = seq_len
        self.use_attention = use_attention
        self.use_edge_enhancement = use_edge_enhancement
        
        # LSTM模块
        self.lstm_modules = nn.ModuleList([
            ConvLSTM(input_dim=c, hidden_dim=c, kernel_size=lstm_kernel)
            for c in ch
        ])
        
        # 注意力模块
        if use_attention:
            self.attention_modules = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(c, c//8, 1),
                    nn.BatchNorm2d(c//8),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(c//8, c, 1),
                    nn.Sigmoid()
                ) for c in ch
            ])
            
        # 边缘增强模块
        if use_edge_enhancement:
            self.edge_enhancement = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(c, c, 3, padding=1),
                    nn.BatchNorm2d(c),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(c, c, 3, padding=1),
                    nn.BatchNorm2d(c),
                    nn.ReLU(inplace=True)
                ) for c in ch
            ])
            
        self._init_state = None
        
    def init_hidden(self, batch_size, spatial_shapes, device=None):
        """初始化LSTM的隐藏状态"""
        state = []
        for i, c in enumerate(self.lstm_modules):
            H, W = spatial_shapes[i]
            h = torch.zeros(batch_size, c.hidden_dim, H, W, device=device)
            c_ = torch.zeros(batch_size, c.hidden_dim, H, W, device=device)
            state.append((h, c_))
        return state
        
    def forward(self, x, state=None):
        """
        前向传播函数
        
        Args:
            x: 输入特征图列表 [B, C, H, W]
            state: LSTM的隐藏状态
            
        Returns:
            (tuple): 包含检测输出和分割输出的元组
        """
        # 将输入转换为序列形式 [B, T, C, H, W]
        x_seq = [feat.unsqueeze(1).repeat(1, self.seq_len, 1, 1, 1) for feat in x]
        
        B = x[0].shape[0]
        device = x[0].device
        
        # 初始化LSTM状态
        if state is None:
            spatial_shapes = [(f.shape[-2], f.shape[-1]) for f in x]
            state = self.init_hidden(B, spatial_shapes, device)
            
        out_seq = []
        new_state = []
        
        # 处理每个尺度的特征
        for i, (feat_seq, lstm, (h, c)) in enumerate(zip(x_seq, self.lstm_modules, state)):
            # LSTM处理时序特征
            out, (h_new, c_new) = lstm(feat_seq, (h, c))
            
            # 应用注意力机制
            if self.use_attention:
                attn = self.attention_modules[i](out[:, -1])
                out = out * attn
                
            # 应用边缘增强
            if self.use_edge_enhancement:
                out = self.edge_enhancement[i](out)
                
            out_seq.append(out[:, -1])  # 只取最后一帧
            new_state.append((h_new, c_new))
            
        # 调用父类的forward方法进行检测和分割
        return super().forward(out_seq), new_state
        
    def get_edge_loss(self, pred, target):
        """计算边缘损失"""
        if not self.use_edge_enhancement:
            return torch.tensor(0.0, device=pred.device)
            
        # Sobel算子
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                             dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                             dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)
        
        # 计算边缘
        pred_edge_x = F.conv2d(pred, sobel_x, padding=1)
        pred_edge_y = F.conv2d(pred, sobel_y, padding=1)
        pred_edge = torch.sqrt(pred_edge_x**2 + pred_edge_y**2)
        
        target_edge_x = F.conv2d(target, sobel_x, padding=1)
        target_edge_y = F.conv2d(target, sobel_y, padding=1)
        target_edge = torch.sqrt(target_edge_x**2 + target_edge_y**2)
        
        # 计算边缘损失
        edge_loss = F.mse_loss(pred_edge, target_edge)
        return edge_loss 