import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CustomSequential(nn.Sequential):
    """
    支持多参数传递的 Sequential 模块
    示例：输出 = CustomSequential(x, t)
    """
    def forward(self, x, *args, **kwargs):
        for module in self:
            if isinstance(module, ResidualBlock):
                x = module(x, *args, **kwargs)  # 传递额外参数给 ResidualBlock
            else:
                x = module(x)  # 其他层仅传递 x
        return x

class SinusoidalPositionEmbeddings(nn.Module):
    """正弦位置编码层，用于时间步嵌入"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings

class ResidualBlock(nn.Module):
    """3D残差块"""
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        
    def forward(self, x, t):
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv2(h)
        h = self.norm2(h)
        
        # 时间嵌入处理
        t_emb = self.time_mlp(F.silu(t)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        h = h + t_emb
        return h + x  # 残差连接

class SelfAttention3D(nn.Module):
    """3D Self-Attention Layer"""
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv3d(channels, channels // 8, 1)
        self.key = nn.Conv3d(channels, channels // 8, 1)
        self.value = nn.Conv3d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch, channels, depth, height, width = x.size()
        query = self.query(x).view(batch, -1, depth * height * width).permute(0, 2, 1)
        key = self.key(x).view(batch, -1, depth * height * width)
        attention = torch.softmax(torch.bmm(query, key), dim=-1)
        value = self.value(x).view(batch, -1, depth * height * width)
        out = torch.bmm(value, attention.permute(0, 2, 1)).view(batch, channels, depth, height, width)
        return self.gamma * out + x

class UNet3D(nn.Module):
    """3D UNet网络结构"""
    def __init__(self, in_channels=1, time_emb_dim=256):
        super().__init__()
        # 时间嵌入层
        self.time_mlp = CustomSequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU()
        )
        
        # 下采样路径
        self.down1 = CustomSequential(
            ResidualBlock(in_channels, 32, time_emb_dim),
            ResidualBlock(32, 32, time_emb_dim),
            SelfAttention3D(32)
        )
        self.down2 = CustomSequential(
            nn.Conv3d(32, 64, 3, stride=2, padding=1),  # 下采样
            ResidualBlock(64, 64, time_emb_dim),
            SelfAttention3D(64)
        )
        self.down3 = CustomSequential(
            nn.Conv3d(64, 128, 3, stride=2, padding=1),  # 下采样
            ResidualBlock(128, 128, time_emb_dim),
            SelfAttention3D(128)
        )
        # 添加第4层下采样
        self.down4 = CustomSequential(
            nn.Conv3d(128, 256, 3, stride=2, padding=1),  # 下采样
            ResidualBlock(256, 256, time_emb_dim),
            SelfAttention3D(256)
        )
        
        # 中间层
        self.middle = CustomSequential(
            ResidualBlock(256, 256, time_emb_dim),
            SelfAttention3D(256),
            ResidualBlock(256, 256, time_emb_dim),
            SelfAttention3D(256)
        )
        
        # 上采样路径
        # 添加第4层上采样
        self.up4 = CustomSequential(
            nn.ConvTranspose3d(256, 128, 4, stride=2, padding=1),  # 上采样
            ResidualBlock(128, 128, time_emb_dim),
            SelfAttention3D(128)
        )
        self.up3 = CustomSequential(
            nn.ConvTranspose3d(128, 64, 4, stride=2, padding=1),  # 上采样
            ResidualBlock(64, 64, time_emb_dim),
            SelfAttention3D(64)
        )
        self.up2 = CustomSequential(
            nn.ConvTranspose3d(64, 32, 4, stride=2, padding=1),
            ResidualBlock(32, 32, time_emb_dim),
            SelfAttention3D(32)
        )
        self.up1 = CustomSequential(
            ResidualBlock(32, 32, time_emb_dim),
            SelfAttention3D(32),
            nn.Conv3d(32, in_channels, 3, padding=1)  # 输出层
        )
        
    def forward(self, x, t):
        time_emb = self.time_mlp(t)
        
        # 下采样路径
        down1 = self.down1(x, time_emb)
        down2 = self.down2(down1, time_emb)
        down3 = self.down3(down2, time_emb)
        down4 = self.down4(down3, time_emb)  # 第4层下采样
        
        # 中间层
        x = self.middle(down4, time_emb)
        
        # 上采样路径 + Skip Connection
        x = self.up4(x, time_emb) + down3  # 第4层上采样
        x = self.up3(x, time_emb) + down2
        x = self.up2(x, time_emb) + down1
        x = self.up1(x, time_emb)
        
        return x