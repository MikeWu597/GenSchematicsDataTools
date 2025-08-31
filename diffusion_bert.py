import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import traceback

class DiffusionModelWithText:
    def __init__(self, model, betas=(1e-4, 0.02), device="cuda"):
        """
        带文本条件的扩散模型初始化
        :param model: UNet3DWithText模型
        :param betas: 扩散步长参数
        :param device: 设备（cuda/cpu）
        """
        print(f"初始化DiffusionModelWithText，设备: {device}，beta范围: {betas}")
        try:
            self.model = model.to(device)
            self.device = device
            
            # Beta调度（线性增加）
            self.beta_start, self.beta_end = betas
            self.T = 1000  # 时间步总数
            print(f"时间步总数: {self.T}")
            
            # 预计算扩散参数
            self.betas = torch.linspace(betas[0], betas[1], self.T).to(device)
            self.alphas = 1. - self.betas
            self.alpha_bars = torch.cumprod(self.alphas, dim=0)
            print("扩散参数预计算完成")
        except Exception as e:
            print(f"DiffusionModelWithText初始化失败: {e}")
            print(f"错误详情: {traceback.format_exc()}")
            raise e
        
    def forward_diffusion(self, x_0, t):
        """
        前向扩散过程：给体素逐步添加噪声
        :param x_0: 原始体素 [B,1,D,H,W]
        :param t: 时间步 [B]
        :return: 加噪后的体素和噪声
        """
        try:
            print(f"执行前向扩散，x_0形状: {x_0.shape}，t形状: {t.shape}")
            alpha_bar = self.alpha_bars[t].view(-1, 1, 1, 1, 1)
            noise = torch.randn_like(x_0)
            x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * noise
            print(f"前向扩散完成，x_t形状: {x_t.shape}")
            return x_t, noise
        except Exception as e:
            print(f"forward_diffusion执行失败: {e}")
            print(f"x_0形状: {x_0.shape}，t形状: {t.shape}")
            print(f"错误详情: {traceback.format_exc()}")
            raise e
    
    def train_step(self, batch, optimizer):
        """
        单步训练：前向扩散 + 反向传播
        :param batch: 包含体素和文本的批次数据
        :param optimizer: 优化器
        :return: 当前损失值
        """
        try:
            print("开始训练步骤")
            self.model.train()
            x_0 = batch['voxels'].to(self.device)
            texts = batch['texts']
            print(f"数据加载完成，体素形状: {x_0.shape}，文本数量: {len(texts)}")
            t = torch.randint(0, self.T, (x_0.shape[0],)).to(self.device)
            print(f"时间步生成完成，t形状: {t.shape}")
            
            # 前向扩散
            x_t, noise = self.forward_diffusion(x_0, t)
            
            # 预测噪声
            predicted_noise = self.model(x_t, t, texts)
            print(f"噪声预测完成，预测噪声形状: {predicted_noise.shape}")
            
            # 计算损失
            loss = F.mse_loss(predicted_noise, noise)
            print(f"损失计算完成，损失值: {loss.item()}")
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print("训练步骤完成")
            return loss.item()
        except Exception as e:
            print(f"训练步骤执行失败: {e}")
            print(f"错误详情: {traceback.format_exc()}")
            raise e
    
    @torch.no_grad()
    def sample(self, text_prompt, shape=(1, 1, 32, 32, 32)):
        """
        根据文本提示生成样本
        :param text_prompt: 文本提示
        :param shape: 输出形状
        :return: 生成的体素数组
        """
        try:
            print(f"开始采样过程，文本提示: {text_prompt}，目标形状: {shape}")
            self.model.eval()
            x = torch.randn(shape).to(self.device)
            print(f"初始噪声生成完成，形状: {x.shape}")
            
            # 重复文本提示以匹配批次大小
            if isinstance(text_prompt, str):
                texts = [text_prompt] * shape[0]
            else:
                texts = text_prompt
            print(f"文本提示处理完成，数量: {len(texts)}")
                
            for i, t in enumerate(reversed(range(self.T))):
                if i % 100 == 0:  # 每100步打印一次进度
                    print(f"采样进度: {i}/{self.T} 步")
                t_batch = torch.full((x.shape[0],), t, device=self.device)
                predicted_noise = self.model(x, t_batch, texts)
                
                # 去噪公式（简化版）
                alpha = self.alphas[t]
                alpha_bar = self.alpha_bars[t]
                beta = self.betas[t]
                
                noise_factor = (1 - alpha) / torch.sqrt(1 - alpha_bar)
                x = (x - noise_factor * predicted_noise) / torch.sqrt(alpha)
                
                if t > 0:
                    noise = torch.randn_like(x)
                    x += torch.sqrt(beta) * noise
                    
            # 转换回二值体素
            x = torch.sigmoid(x)
            result = (x > 0.5).float()
            print(f"采样完成，结果形状: {result.shape}")
            return result
        except Exception as e:
            print(f"采样过程失败: {e}")
            print(f"文本提示: {text_prompt}，目标形状: {shape}")
            print(f"错误详情: {traceback.format_exc()}")
            raise e