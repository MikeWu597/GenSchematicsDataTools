import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import BertTokenizer, BertModel
import os
import traceback

# 获取当前运行目录
current_dir = os.getcwd()

# 拼接本地路径
DEFAULT_BERT_PATH = os.path.join(current_dir, "checkpoints", "bert-base-chinese")
DEFAULT_NON_BERT_MODEL_PATH = os.path.join(current_dir, "checkpoints", "diffusion_20250824_1927_epoch200.pt")

print(f"当前工作目录: {current_dir}")
print(f"默认BERT路径: {DEFAULT_BERT_PATH}")
print(f"默认非BERT模型路径: {DEFAULT_NON_BERT_MODEL_PATH}")


class CustomSequential(nn.Sequential):
    """
    支持多参数传递的 Sequential 模块
    示例：输出 = CustomSequential(x, t, text_emb)
    """
    def forward(self, x, *args, **kwargs):
        for i, module in enumerate(self):
            try:
                if isinstance(module, (ResidualBlock, SelfAttention3D)):
                    x = module(x, *args, **kwargs)  # 传递额外参数
                else:
                    x = module(x)  # 其他层仅传递 x
            except Exception as e:
                print(f"CustomSequential中第 {i} 个模块执行出错: {e}")
                print(f"错误详情: {traceback.format_exc()}")
                raise e
        return x

class SinusoidalPositionEmbeddings(nn.Module):
    """正弦位置编码层，用于时间步嵌入"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        print(f"初始化SinusoidalPositionEmbeddings，维度: {dim}")
        
    def forward(self, time):
        try:
            device = time.device
            half_dim = self.dim // 2
            embeddings = np.log(10000) / (half_dim - 1)
            embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
            embeddings = time[:, None] * embeddings[None, :]
            embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
            return embeddings
        except Exception as e:
            print(f"SinusoidalPositionEmbeddings forward出错: {e}")
            print(f"输入time形状: {time.shape}")
            print(f"错误详情: {traceback.format_exc()}")
            raise e

class TextEncoder(nn.Module):
    """BERT文本编码器，支持中文BERT-wwm"""
    def __init__(self, model_name='hfl/chinese-bert-wwm-ext', freeze_bert=True):
        super().__init__()
        self.model_name = DEFAULT_BERT_PATH if DEFAULT_BERT_PATH else model_name
        self.freeze_bert = freeze_bert
        
        print(f"初始化TextEncoder，模型名: {self.model_name}，冻结BERT: {freeze_bert}")
        
        try:
            print(f"尝试从本地加载BERT分词器: {self.model_name}")
            self.tokenizer = BertTokenizer.from_pretrained(self.model_name, local_files_only=True)
            print("BERT分词器加载成功")
        except Exception as e:
            print(f"加载BERT分词器失败: {e}")
            print("尝试从HuggingFace加载...")
            try:
                self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
                print("从HuggingFace加载BERT分词器成功")
            except Exception as e2:
                print(f"从HuggingFace加载BERT分词器也失败了: {e2}")
                print(f"错误详情: {traceback.format_exc()}")
                raise e2
        
        try:
            print(f"尝试从本地加载BERT模型: {self.model_name}")
            self.bert = BertModel.from_pretrained(self.model_name, local_files_only=True)
            print("BERT模型加载成功")
        except Exception as e:
            print(f"加载BERT模型失败: {e}")
            print("尝试从HuggingFace加载...")
            try:
                self.bert = BertModel.from_pretrained(self.model_name)
                print("从HuggingFace加载BERT模型成功")
            except Exception as e2:
                print(f"从HuggingFace加载BERT模型也失败了: {e2}")
                print(f"错误详情: {traceback.format_exc()}")
                raise e2
                
        if freeze_bert:
            # 冻结BERT参数以避免微调
            frozen_count = 0
            for param in self.bert.parameters():
                param.requires_grad = False
                frozen_count += 1
            print(f"已冻结BERT参数，共冻结 {frozen_count} 个参数")
                
        # 添加投影层以匹配时间嵌入维度
        try:
            self.projection = nn.Linear(self.bert.config.hidden_size, 256)
            print(f"创建投影层，输入维度: {self.bert.config.hidden_size}，输出维度: 256")
        except Exception as e:
            print(f"创建投影层失败: {e}")
            print(f"错误详情: {traceback.format_exc()}")
            raise e
        
    def forward(self, text):
        """
        对文本进行编码
        :param text: 文本列表
        :return: 文本嵌入 [B, text_emb_dim]
        """
        try:
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
                print(f"文本列表tokenization完成，批次大小: {len(text)}")
            else:
                # 如果是单个文本
                tokens = self.tokenizer(
                    text, 
                    return_tensors='pt', 
                    padding=True, 
                    truncation=True, 
                    max_length=64
                ).to(next(self.bert.parameters()).device)
                print(f"单个文本tokenization完成: {text}")
            
            # 获取BERT输出
            outputs = self.bert(**tokens)
            # 使用[CLS]标记的表示
            text_embeddings = outputs.last_hidden_state[:, 0, :]  # [B, hidden_size]
            print(f"BERT输出形状: {text_embeddings.shape}")
            # 投影到目标维度
            text_embeddings = self.projection(text_embeddings)  # [B, time_emb_dim]
            print(f"投影后形状: {text_embeddings.shape}")
            return text_embeddings
        except Exception as e:
            print(f"TextEncoder forward出错: {e}")
            print(f"输入文本: {text}")
            print(f"错误详情: {traceback.format_exc()}")
            raise e

class ResidualBlock(nn.Module):
    """3D残差块，支持文本条件"""
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        print(f"初始化ResidualBlock，输入通道: {in_channels}，输出通道: {out_channels}，时间嵌入维度: {time_emb_dim}")
        try:
            self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
            self.norm1 = nn.GroupNorm(8, out_channels)
            self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
            self.norm2 = nn.GroupNorm(8, out_channels)
            self.time_mlp = nn.Linear(time_emb_dim, out_channels)
            # 添加文本条件MLP
            self.text_mlp = nn.Linear(time_emb_dim, out_channels)
            # print("ResidualBlock各层初始化完成")
        except Exception as e:
            print(f"ResidualBlock初始化失败: {e}")
            print(f"错误详情: {traceback.format_exc()}")
            raise e
        
    def forward(self, x, t, text_emb=None):
        try:
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
                # print(f"ResidualBlock处理完成，包含文本条件，输出形状: {h.shape}")
            else:
                # print(f"ResidualBlock处理完成，无文本条件，输出形状: {h.shape}")
                pass
                
            return h + x  # 残差连接
        except Exception as e:
            print(f"ResidualBlock forward出错: {e}")
            print(f"输入x形状: {x.shape}, t形状: {t.shape}")
            print(f"是否有文本嵌入: {text_emb is not None}")
            if text_emb is not None:
                print(f"文本嵌入形状: {text_emb.shape}")
            print(f"错误详情: {traceback.format_exc()}")
            raise e

class SelfAttention3D(nn.Module):
    """3D自注意力机制"""
    def __init__(self, channels, text_emb_dim=256):
        super().__init__()
        print(f"初始化SelfAttention3D，通道数: {channels}，文本嵌入维度: {text_emb_dim}")
        try:
            self.query = nn.Conv3d(channels, channels // 8, 1)
            self.key = nn.Conv3d(channels, channels // 8, 1)
            self.value = nn.Conv3d(channels, channels, 1)
            self.gamma = nn.Parameter(torch.zeros(1))
            
            # 添加文本条件投影层
            self.text_key_proj = nn.Linear(text_emb_dim, channels // 8)
            self.text_value_proj = nn.Linear(text_emb_dim, channels)
            print("SelfAttention3D各层初始化完成")
        except Exception as e:
            print(f"SelfAttention3D初始化失败: {e}")
            print(f"错误详情: {traceback.format_exc()}")
            raise e

    def forward(self, x, t=None, text_emb=None):
        try:
            batch, channels, depth, height, width = x.size()
            # print(f"SelfAttention3D处理，输入形状: B={batch}, C={channels}, D={depth}, H={height}, W={width}")
            
            # 如果提供了文本嵌入，则使用交叉注意力
            if text_emb is not None:
                # print("使用交叉注意力机制")
                # 投影文本嵌入
                text_key = self.text_key_proj(text_emb)   # [B, channels // 8]
                text_value = self.text_value_proj(text_emb)  # [B, channels]
                # print(f"文本嵌入投影完成，text_key形状: {text_key.shape}，text_value形状: {text_value.shape}")
                
                # 重塑文本嵌入以匹配注意力计算
                text_key = text_key.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, depth, height, width)
                text_value = text_value.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, depth, height, width)
                # print(f"文本嵌入重塑完成，text_key形状: {text_key.shape}，text_value形状: {text_value.shape}")
                
                # 计算query
                query = self.query(x).view(batch, -1, depth * height * width).permute(0, 2, 1)
                # print(f"query计算完成，形状: {query.shape}")
                
                # 计算key并结合文本信息
                img_key = self.key(x).view(batch, -1, depth * height * width)
                text_key = text_key.view(batch, -1, depth * height * width)
                combined_key = img_key + text_key
                # print(f"key计算完成，img_key形状: {img_key.shape}，combined_key形状: {combined_key.shape}")
                
                # 计算注意力权重
                attention = torch.softmax(torch.bmm(query, combined_key), dim=-1)
                # print(f"注意力权重计算完成，形状: {attention.shape}")
                
                # 计算value并结合文本信息
                img_value = self.value(x).view(batch, -1, depth * height * width)
                text_value = text_value.view(batch, -1, depth * height * width)
                combined_value = img_value + text_value
                # print(f"value计算完成，img_value形状: {img_value.shape}，combined_value形状: {combined_value.shape}")
                
                # 应用注意力权重
                out = torch.bmm(combined_value, attention.permute(0, 2, 1)).view(batch, channels, depth, height, width)
                # print(f"注意力应用完成，输出形状: {out.shape}")
            else:
                print("使用标准自注意力机制")
                # 标准自注意力机制
                query = self.query(x).view(batch, -1, depth * height * width).permute(0, 2, 1)
                key = self.key(x).view(batch, -1, depth * height * width)
                attention = torch.softmax(torch.bmm(query, key), dim=-1)
                value = self.value(x).view(batch, -1, depth * height * width)
                out = torch.bmm(value, attention.permute(0, 2, 1)).view(batch, channels, depth, height, width)
                # print(f"标准自注意力完成，输出形状: {out.shape}")
                
            result = self.gamma * out + x
            # print(f"SelfAttention3D处理完成，最终输出形状: {result.shape}")
            return result
        except Exception as e:
            print(f"SelfAttention3D forward出错: {e}")
            print(f"输入x形状: {x.shape}")
            print(f"是否有时间信息: {t is not None}")
            print(f"是否有文本嵌入: {text_emb is not None}")
            if text_emb is not None:
                print(f"文本嵌入形状: {text_emb.shape}")
            print(f"错误详情: {traceback.format_exc()}")
            raise e

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
            SelfAttention3D(32, time_emb_dim)  # 添加自注意力
        )
        self.down2 = CustomSequential(
            nn.Conv3d(32, 128, 3, stride=2, padding=1),  # 下采样
            ResidualBlock(128, 128, time_emb_dim),
            SelfAttention3D(128, time_emb_dim)  # 添加自注意力
        )
        self.down3 = CustomSequential(
            nn.Conv3d(128, 256, 3, stride=2, padding=1),
            ResidualBlock(256, 256, time_emb_dim),
            SelfAttention3D(256, time_emb_dim)  # 添加自注意力
        )
        
        # 中间层
        self.middle = CustomSequential(
            ResidualBlock(256, 256, time_emb_dim),
            SelfAttention3D(256, time_emb_dim),  # 添加自注意力
            ResidualBlock(256, 256, time_emb_dim),
            SelfAttention3D(256, time_emb_dim)  # 添加自注意力
        )
        
        # 上采样路径
        self.up3 = CustomSequential(
            nn.ConvTranspose3d(256, 128, 4, stride=2, padding=1),  # 上采样
            ResidualBlock(128, 128, time_emb_dim),
            SelfAttention3D(128, time_emb_dim)  # 添加自注意力
        )
        self.up2 = CustomSequential(
            nn.ConvTranspose3d(128, 32, 4, stride=2, padding=1),
            ResidualBlock(32, 32, time_emb_dim),
            SelfAttention3D(32, time_emb_dim)  # 添加自注意力
        )
        self.up1 = CustomSequential(
            ResidualBlock(32, 32, time_emb_dim),
            SelfAttention3D(32, time_emb_dim),  # 添加自注意力
            nn.Conv3d(32, in_channels, 3, padding=1)  # 输出层
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
            SelfAttention3D(32, time_emb_dim)  # 添加自注意力
        )
        self.down2 = CustomSequential(
            nn.Conv3d(32, 128, 3, stride=2, padding=1),  # 下采样
            ResidualBlock(128, 128, time_emb_dim),
            SelfAttention3D(128, time_emb_dim)  # 添加自注意力
        )
        self.down3 = CustomSequential(
            nn.Conv3d(128, 256, 3, stride=2, padding=1),
            ResidualBlock(256, 256, time_emb_dim),
            SelfAttention3D(256, time_emb_dim)  # 添加自注意力
        )
        
        # 中间层
        self.middle = CustomSequential(
            ResidualBlock(256, 256, time_emb_dim),
            SelfAttention3D(256, time_emb_dim),  # 添加自注意力
            ResidualBlock(256, 256, time_emb_dim),
            SelfAttention3D(256, time_emb_dim)  # 添加自注意力
        )
        
        # 上采样路径
        self.up3 = CustomSequential(
            nn.ConvTranspose3d(256, 128, 4, stride=2, padding=1),  # 上采样
            ResidualBlock(128, 128, time_emb_dim),
            SelfAttention3D(128, time_emb_dim)  # 添加自注意力
        )
        self.up2 = CustomSequential(
            nn.ConvTranspose3d(128, 32, 4, stride=2, padding=1),
            ResidualBlock(32, 32, time_emb_dim),
            SelfAttention3D(32, time_emb_dim)  # 添加自注意力
        )
        self.up1 = CustomSequential(
            ResidualBlock(32, 32, time_emb_dim),
            SelfAttention3D(32, time_emb_dim),  # 添加自注意力
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
