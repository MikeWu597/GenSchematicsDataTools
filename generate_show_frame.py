import numpy as np
import torch
from datetime import datetime
from model import UNet3D
from diffusion_show_frame import DiffusionModel
from tqdm import tqdm
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # 自动选择设备

def load_model(checkpoint_path):
    """加载训练好的模型"""
    model = UNet3D()
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def generate_and_save(model_path, output_dir="generated"):
    """
    生成样本并保存为npy文件
    :param model_path: 模型路径
    :param output_dir: 输出目录
    """
    model = load_model(model_path)
    diffusion = DiffusionModel(model, device=DEVICE)
    
    os.makedirs(output_dir, exist_ok=True)
    sample = diffusion.sample()
    
    # 转换为numpy数组
    voxel_array = sample.squeeze().cpu().numpy() 
    
    # 保存为npy文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    np.save(os.path.join(output_dir, f"generated_{timestamp}.npy"), voxel_array)
    
    print(f"生成完成，体素数据已保存至 {output_dir}")
    return voxel_array

# 示例调用
if __name__ == "__main__":
    generated_voxel = generate_and_save("checkpoints/diffusion_20250823_2112_epoch200.pt")
