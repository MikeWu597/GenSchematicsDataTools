import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import traceback
from eval_metrics import chamfer_distance, voxel_to_point_cloud

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
    
    def train_step(self, batch, optimizer, gradient_clip_value=1.0, alpha=0.7, beta=0.3):
        """
        单步训练：前向扩散 + 反向传播
        :param batch: 包含体素和文本的批次数据
        :param optimizer: 优化器
        :param gradient_clip_value: 梯度裁剪阈值
        :param alpha: MSE损失权重
        :param beta: Chamfer Distance损失权重
        :return: 当前损失值
        """
        try:
            # print("开始训练步骤")
            self.model.train()
            x_0 = batch['voxels'].to(self.device)
            texts = batch['texts']
            # print(f"数据加载完成，体素形状: {x_0.shape}，文本数量: {len(texts)}")
            t = torch.randint(0, self.T, (x_0.shape[0],)).to(self.device)
            # print(f"时间步生成完成，t形状: {t.shape}")
            
            # 前向扩散
            x_t, noise = self.forward_diffusion(x_0, t)
            
            # 预测噪声
            predicted_noise = self.model(x_t, t, texts)
            # print(f"噪声预测完成，预测噪声形状: {predicted_noise.shape}")
            
            # 计算MSE损失
            loss_mse = F.mse_loss(predicted_noise, noise)
            
            # 计算Chamfer Distance损失
            # 通过模型预测去噪后的体素
            with torch.no_grad():
                # 使用单步去噪来获取预测的原始体素
                alpha_t = self.alphas[t].view(-1, 1, 1, 1, 1)
                alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1, 1, 1)
                pred_x0 = (x_t - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)
                pred_x0 = torch.sigmoid(pred_x0)
                
            # 将体素转换为点云并计算Chamfer Distance
            loss_cd = 0
            batch_size = x_0.shape[0]
            for i in range(batch_size):
                # 将真实体素和预测体素转换为点云
                gt_points = voxel_to_point_cloud(x_0[i].squeeze())
                pred_points = voxel_to_point_cloud(pred_x0[i].squeeze())
                
                # 只有当两个点云都非空时才计算Chamfer Distance
                if len(gt_points) > 0 and len(pred_points) > 0:
                    gt_points_tensor = torch.from_numpy(gt_points).float().to(self.device).unsqueeze(0)
                    pred_points_tensor = torch.from_numpy(pred_points).float().to(self.device).unsqueeze(0)
                    cd = chamfer_distance(gt_points_tensor, pred_points_tensor)
                    loss_cd += cd.mean()
                elif len(gt_points) > 0 or len(pred_points) > 0:
                    # 如果一个为空另一个非空，增加损失
                    loss_cd += torch.tensor(1.0, device=self.device)
            
            loss_cd = loss_cd / batch_size if batch_size > 0 else torch.tensor(0.0, device=self.device)
            
            # 组合损失
            total_loss = alpha * loss_mse + beta * loss_cd
            
            # print(f"损失计算完成，MSE损失: {loss_mse.item()}, CD损失: {loss_cd.item()}, 总损失: {total_loss.item()}")
            
            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            
            # 梯度裁剪，防止梯度爆炸和梯度消失
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip_value)
            
            optimizer.step()
            
            # print("训练步骤完成")
            return total_loss.item()
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