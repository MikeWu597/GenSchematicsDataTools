import numpy as np
import torch
from datetime import datetime
from model_bert import UNet3DHybrid, DEFAULT_BERT_PATH
from diffusion_bert import DiffusionModelWithText
import os
import traceback
import sys

# 默认模型路径配置
DEFAULT_CHECKPOINT_PATH = "checkpoints_bert/diffusion_bert_hybrid_20250824_1927_epoch200.pt"  # 默认Minecraft模型路径

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # 自动选择设备

def load_model(checkpoint_path):
    """加载训练好的模型"""
    print(f"开始加载模型，设备: {DEVICE}")
    
    try:
        model = UNet3DHybrid()
        print("UNet3DHybrid模型初始化完成")
    except Exception as e:
        print(f"模型初始化失败: {e}")
        print(f"详细错误信息: {traceback.format_exc()}")
        raise e
    
    # 直接从融合模型路径加载权重
    if os.path.exists(checkpoint_path):
        try:
            print(f"尝试加载融合模型权重从 {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"成功加载融合模型权重从 {checkpoint_path}")
        except Exception as e:
            print(f"加载融合模型权重失败: {e}")
    else:
        print(f"警告: 融合模型文件 {checkpoint_path} 不存在")
        print("请检查模型文件路径是否正确")
    
    return model

def generate_and_save(model_path, text_prompt, output_dir="generated_bert"):
    """
    根据文本提示生成样本并保存为npy文件
    :param model_path: 模型路径
    :param text_prompt: 文本提示
    :param output_dir: 输出目录
    """
    print(f"开始生成样本，文本提示: {text_prompt}")
    model = load_model(model_path)
    diffusion = DiffusionModelWithText(model, device=DEVICE)
    
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        print(f"创建输出目录失败 {output_dir}: {e}")
        print(f"详细错误信息: {traceback.format_exc()}")
        return None
    print(f"开始采样过程")
    print(f"开始采样过程")
    try:
        sample = diffusion.sample(text_prompt)
        print(f"采样完成")
    except Exception as e:
        print(f"采样过程失败: {e}")
        print(f"详细错误信息: {traceback.format_exc()}")
        return None
    
    # 转换为numpy数组
    # 转换为numpy数组
    try:
        voxel_array = sample.squeeze().cpu().numpy() 
    except Exception as e:
        print(f"转换为numpy数组失败: {e}")
        print(f"详细错误信息: {traceback.format_exc()}")
        return None
    
    # 保存为npy文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"generated_{timestamp}_{text_prompt.replace(' ', '_')}.npy"
    # 保存为npy文件
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        filename = f"generated_{timestamp}_{text_prompt.replace(' ', '_')}.npy"
        save_path = os.path.join(output_dir, filename)
        np.save(save_path, voxel_array)
        print(f"生成完成，体素数据已保存至 {save_path}")
    except Exception as e:
        print(f"保存文件失败: {e}")
        print(f"详细错误信息: {traceback.format_exc()}")
        return None
    
    return voxel_array

# 示例调用
if __name__ == "__main__":
    print("启动生成脚本")
    # 示例文本提示
    text_prompt = "a house with a red roof"
    print(f"使用文本提示: {text_prompt}")
    generated_voxel = generate_and_save(
        DEFAULT_CHECKPOINT_PATH,
        text_prompt
    )