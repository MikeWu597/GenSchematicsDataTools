import numpy as np
import torch
from datetime import datetime
from model_bert import UNet3DHybrid, DEFAULT_BERT_PATH
from diffusion_bert import DiffusionModelWithText
import os

# 默认模型路径配置
DEFAULT_CHECKPOINT_PATH = "checkpoints_bert/diffusion_bert_hybrid_20250824_1927_epoch200.pt"  # 默认Minecraft模型路径
BERT_PATH = None  # 如果设置为本地BERT模型路径，则使用本地BERT模型（会覆盖model_bert.py中的DEFAULT_BERT_PATH）

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # 自动选择设备

def load_model(checkpoint_path):
    """加载训练好的模型"""
    # 如果指定了本地BERT路径，则更新DEFAULT_BERT_PATH
    if BERT_PATH:
        import model_bert
        model_bert.DEFAULT_BERT_PATH = BERT_PATH
    
    model = UNet3DHybrid()
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def generate_and_save(model_path, text_prompt, output_dir="generated_bert"):
    """
    根据文本提示生成样本并保存为npy文件
    :param model_path: 模型路径
    :param text_prompt: 文本提示
    :param output_dir: 输出目录
    """
    model = load_model(model_path)
    diffusion = DiffusionModelWithText(model, device=DEVICE)
    
    os.makedirs(output_dir, exist_ok=True)
    sample = diffusion.sample(text_prompt)
    
    # 转换为numpy数组
    voxel_array = sample.squeeze().cpu().numpy() 
    
    # 保存为npy文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"generated_{timestamp}_{text_prompt.replace(' ', '_')}.npy"
    np.save(os.path.join(output_dir, filename), voxel_array)
    
    print(f"生成完成，体素数据已保存至 {os.path.join(output_dir, filename)}")
    return voxel_array

# 示例调用
if __name__ == "__main__":
    # 示例文本提示
    text_prompt = "a house with a red roof"
    generated_voxel = generate_and_save(
        DEFAULT_CHECKPOINT_PATH,
        text_prompt
    )