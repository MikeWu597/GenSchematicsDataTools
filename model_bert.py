import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import BertTokenizer, BertModel

# 默认BERT模型路径配置
DEFAULT_BERT_PATH = "checkpoints/chinese-bert-wwm-ext"  # 如果设置为本地路径，则加载本地BERT模型
DEFAULT_NON_BERT_MODEL_PATH = "checkpoints/diffusion_20250824_1927_epoch200.pt"  # 无约束模型路径配置

class CustomSequential(nn.Sequential):
    """
    支持多参数传递的 Sequential 模块
    示例：输出 = CustomSequential(x, t, text_emb)
    """
    def forward(self, x, *args, **kwargs):
        for module in self:
            if isinstance(module, (ResidualBlock, SelfAttention3D)):
                x = module(x, *args, **kwargs)  # 传递额外参数
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

class TextEncoder(nn.Module):
    """BERT文本编码器，支持中文BERT-wwm"""
    def __init__(self, model_name='hfl/chinese-bert-wwm-ext', freeze_bert=True):
        super().__init__()
        self.model_name = DEFAULT_BERT_PATH if DEFAULT_BERT_PATH else model_name
        
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.bert = BertModel.from_pretrained(self.model_name)
        
        if freeze_bert:
            # 冻结BERT参数以避免微调
            for param in self.bert.parameters():
                param.requires_grad = False
                
        # 添加投影层以匹配时间嵌入维度
        self.projection = nn.Linear(self.bert.config.hidden_size, 256)
        
    def forward(self, text):
        """
        对文本进行编码
        :param text: 文本列表
        :return: 文本嵌入 [B, text_emb_dim]
        """
        # 对文本进行tokenization
        if isinstance(text, list):
            # 如果是文本列表，进行批处理
            tokens = self.tokenizer(
                text, 
                return_tensors='pt', 
                padding=True, 
                truncation=True, 
                max_length=64
            ).to(next(self.bert.parameters()).device)
        else:
            # 如果是单个文本
            tokens = self.tokenizer(
                text, 
                return_tensors='pt', 
                padding=True, 
                truncation=True, 
                max_length=64
            ).to(next(self.bert.parameters()).device)
            
        # 获取BERT输出
        outputs = self.bert(**tokens)
        # 使用[CLS]标记的表示
        text_embeddings = outputs.last_hidden_state[:, 0, :]  # [B, hidden_size]
        # 投影到目标维度
        text_embeddings = self.projection(text_embeddings)  # [B, time_emb_dim]
        return text_embeddings

class ResidualBlock(nn.Module):
    """3D残差块，支持文本条件"""
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        # 添加文本条件MLP
        self.text_mlp = nn.Linear(time_emb_dim, out_channels)
        
    def forward(self, x, t, text_emb=None):
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv2(h)
        h = self.norm2(h)
        
        # 时间嵌入处理
        t_emb = self.time_mlp(F.silu(t)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        h = h + t_emb
        
        # 文本条件处理
        if text_emb is not None:
            text_cond = self.text_mlp(F.silu(text_emb)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            h = h + text_cond
            
        return h + x  # 残差连接

class SelfAttention3D(nn.Module):
    """3D自注意力机制"""
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv3d(channels, channels // 8, 1)
        self.key = nn.Conv3d(channels, channels // 8, 1)
        self.value = nn.Conv3d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, t=None, text_emb=None):
        batch, channels, depth, height, width = x.size()
        query = self.query(x).view(batch, -1, depth * height * width).permute(0, 2, 1)
        key = self.key(x).view(batch, -1, depth * height * width)
        attention = torch.softmax(torch.bmm(query, key), dim=-1)
        value = self.value(x).view(batch, -1, depth * height * width)
        out = torch.bmm(value, attention.permute(0, 2, 1)).view(batch, channels, depth, height, width)
        return self.gamma * out + x

class UNet3DWithText(nn.Module):
    """带文本条件的3D UNet网络结构"""
    def __init__(self, in_channels=1, time_emb_dim=256, freeze_bert=True):
        super().__init__()
        # 文本编码器
        self.text_encoder = TextEncoder(model_name='hfl/chinese-bert-wwm-ext', freeze_bert=freeze_bert)
        
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
            SelfAttention3D(32)  # 添加自注意力
        )
        self.down2 = CustomSequential(
            nn.Conv3d(32, 128, 3, stride=2, padding=1),  # 下采样
            ResidualBlock(128, 128, time_emb_dim),
            SelfAttention3D(128)  # 添加自注意力
        )
        self.down3 = CustomSequential(
            nn.Conv3d(128, 256, 3, stride=2, padding=1),
            ResidualBlock(256, 256, time_emb_dim),
            SelfAttention3D(256)  # 添加自注意力
        )
        
        # 中间层
        self.middle = CustomSequential(
            ResidualBlock(256, 256, time_emb_dim),
            SelfAttention3D(256),  # 添加自注意力
            ResidualBlock(256, 256, time_emb_dim),
            SelfAttention3D(256)  # 添加自注意力
        )
        
    def forward(self, x, t, text=None):
        # 文本编码
        if text is not None:
            text_emb = self.text_encoder(text)
        else:
            text_emb = None
            
        time_emb = self.time_mlp(t)
        
        # 下采样路径
        down1 = self.down1(x, time_emb, text_emb)
        down2 = self.down2(down1, time_emb, text_emb)
        down3 = self.down3(down2, time_emb, text_emb)
        
        # 中间层
        x = self.middle(down3, time_emb, text_emb)
        
        # 上采样路径 + Skip Connection
        x = self.up3(x, time_emb, text_emb) + down2
        x = self.up2(x, time_emb, text_emb) + down1
        x = self.up1(x, time_emb, text_emb)
        
        return x

class UNet3DHybrid(nn.Module):
    """混合模型，结合了BERT文本条件和无文本条件的模型"""
    def __init__(self, in_channels=1, time_emb_dim=256, freeze_bert=True):
        super().__init__()
        # 文本编码器
        self.text_encoder = TextEncoder(model_name='hfl/chinese-bert-wwm-ext', freeze_bert=freeze_bert)
        
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
            SelfAttention3D(32)  # 添加自注意力
        )
        self.down2 = CustomSequential(
            nn.Conv3d(32, 128, 3, stride=2, padding=1),  # 下采样
            ResidualBlock(128, 128, time_emb_dim),
            SelfAttention3D(128)  # 添加自注意力
        )
        self.down3 = CustomSequential(
            nn.Conv3d(128, 256, 3, stride=2, padding=1),
            ResidualBlock(256, 256, time_emb_dim),
            SelfAttention3D(256)  # 添加自注意力
        )
        
        # 中间层
        self.middle = CustomSequential(
            ResidualBlock(256, 256, time_emb_dim),
            SelfAttention3D(256),  # 添加自注意力
            ResidualBlock(256, 256, time_emb_dim),
            SelfAttention3D(256)  # 添加自注意力
        )
        
        # 上采样路径
        self.up3 = CustomSequential(
            nn.ConvTranspose3d(256, 128, 4, stride=2, padding=1),  # 上采样
            ResidualBlock(128, 128, time_emb_dim),
            SelfAttention3D(128)  # 添加自注意力
        )
        self.up2 = CustomSequential(
            nn.ConvTranspose3d(128, 32, 4, stride=2, padding=1),
            ResidualBlock(32, 32, time_emb_dim),
            SelfAttention3D(32)  # 添加自注意力
        )
        self.up1 = CustomSequential(
            ResidualBlock(32, 32, time_emb_dim),
            SelfAttention3D(32),  # 添加自注意力
            nn.Conv3d(32, in_channels, 3, padding=1)  # 输出层
        )
    
    def load_unconstrained_model_weights(self, model_path=None):
        """加载无约束（非文本条件）模型的权重"""
        if model_path is None and DEFAULT_NON_BERT_MODEL_PATH is None:
            raise ValueError("无约束模型路径未设置，请提供路径")
            
        load_path = model_path if model_path is not None else DEFAULT_NON_BERT_MODEL_PATH
        
        # 加载无约束模型权重
        state_dict = torch.load(load_path)
        
        # 过滤出当前模型需要的权重
        filtered_state_dict = {
            k: v for k, v in state_dict.items() if k in self.state_dict()
        }
        
        # 加载权重
        self.load_state_dict(filtered_state_dict, strict=False)
        print(f"加载无约束模型权重成功: {load_path}")
        
    def forward(self, x, t, text=None):
        # 文本编码
        if text is not None:
            text_emb = self.text_encoder(text)
        else:
            text_emb = None
            
        time_emb = self.time_mlp(t)
        
        # 下采样路径
        down1 = self.down1(x, time_emb, text_emb)
        down2 = self.down2(down1, time_emb, text_emb)
        down3 = self.down3(down2, time_emb, text_emb)
        
        # 中间层
        x = self.middle(down3, time_emb, text_emb)
        
        # 上采样路径 + Skip Connection
        x = self.up3(x, time_emb, text_emb) + down2
        x = self.up2(x, time_emb, text_emb) + down1
        x = self.up1(x, time_emb, text_emb)
        
        return x
