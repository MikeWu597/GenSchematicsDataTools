import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import os

class DiffusionModel:
    def __init__(self, model, betas=(1e-4, 0.02), device="cuda"):
        """
        扩散模型初始化
        :param model: UNet3D模型
        :param betas: 扩散步长参数
        :param device: 设备（cuda/cpu）
        """
        self.model = model.to(device)
        self.device = device
        
        # Beta调度（线性增加）
        self.beta_start, self.beta_end = betas
        self.T = 1000  # 时间步总数
        
        # 预计算扩散参数
        self.betas = torch.linspace(betas[0], betas[1], self.T).to(device)
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        
    def forward_diffusion(self, x_0, t):
        """
        前向扩散过程：给体素逐步添加噪声
        :param x_0: 原始体素 [B,1,D,H,W]
        :param t: 时间步 [B]
        :return: 加噪后的体素和噪声
        """
        alpha_bar = self.alpha_bars[t].view(-1, 1, 1, 1, 1)
        noise = torch.randn_like(x_0)
        x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * noise
        return x_t, noise
    
    def train_step(self, x_0, optimizer):
        """
        单步训练：前向扩散 + 反向传播
        :param x_0: 输入体素
        :param optimizer: 优化器
        :return: 当前损失值
        """
        self.model.train()
        x_0 = x_0.to(self.device)
        t = torch.randint(0, self.T, (x_0.shape[0],)).to(self.device)
        
        # 前向扩散
        x_t, noise = self.forward_diffusion(x_0, t)
        
        # 预测噪声
        predicted_noise = self.model(x_t, t)
        
        # 计算损失
        loss = F.mse_loss(predicted_noise, noise)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def sample(self, shape=(1, 1, 32, 32, 32)):
        """
        从纯噪声生成样本
        :param shape: 输出形状
        :return: 生成的体素数组
        """
        self.model.eval()
        x = torch.randn(shape).to(self.device)
        
        for t in reversed(range(self.T)):
            t_batch = torch.full((x.shape[0],), t, device=self.device)
            predicted_noise = self.model(x, t_batch)
            
            # 去噪公式（简化版）
            alpha = self.alphas[t]
            alpha_bar = self.alpha_bars[t]
            beta = self.betas[t]
            
            noise_factor = (1 - alpha) / torch.sqrt(1 - alpha_bar)
            x = (x - noise_factor * predicted_noise) / torch.sqrt(alpha)
            
            if t > 0:
                noise = torch.randn_like(x)
                x += torch.sqrt(beta) * noise

            if t % 100 == 0:
                print(f"Sampling step {t} completed")
                # Apply the same final transformation to intermediate frames
                # so they are saved in the same format as the final output
                frame_data = torch.sigmoid(x)
                frame_data = (frame_data > 0.5).float()
                frame_data = frame_data.squeeze().cpu().numpy()
                os.makedirs("frames", exist_ok=True)
                frame_path = os.path.join("frames", f"frame_{t}.npy")
                np.save(frame_path, frame_data)
                
        # 转换回二值体素
        x = torch.sigmoid(x)
        return (x > 0.5).float()
