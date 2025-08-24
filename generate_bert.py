import numpy as np
import torch
from datetime import datetime
from model_bert import UNet3DHybrid, DEFAULT_BERT_PATH
from diffusion_bert import DiffusionModelWithText
import os

# 默认模型路径配置
DEFAULT_CHECKPOINT_PATH = "checkpoints_bert/diffusion_bert_hybrid_20250824_1927_epoch200.pt"  # 默认Minecraft模型路径
BERT_PATH = None  # 如果设置为本地BERT模型路径，则使用本地BERT模型（会覆盖model_bert.py中的DEFAULT_BERT_PATH）
NON_BERT_MODEL_PATH = "checkpoints/diffusion_20250824_1927_epoch200.pt"  # 无约束模型路径配置

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # 自动选择设备

def load_model(checkpoint_path):
    """加载训练好的模型"""
    print(f"开始加载模型，设备: {DEVICE}")
    # 如果指定了本地BERT路径，则更新DEFAULT_BERT_PATH
    if BERT_PATH:
        import model_bert
        model_bert.DEFAULT_BERT_PATH = BERT_PATH
        print(f"使用自定义BERT路径: {BERT_PATH}")
    
    model = UNet3DHybrid()
    print("UNet3DHybrid模型初始化完成")
    
    # 如果指定了无约束模型路径，则加载预训练权重
    if NON_BERT_MODEL_PATH and os.path.exists(NON_BERT_MODEL_PATH):
        try:
            print(f"尝试加载无约束模型权重从 {NON_BERT_MODEL_PATH}")
            checkpoint = torch.load(NON_BERT_MODEL_PATH, map_location=DEVICE)
            # 加载匹配的权重
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if k in model_dict and model_dict[k].shape == v.shape}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            print(f"成功加载无约束模型权重从 {NON_BERT_MODEL_PATH}")
        except Exception as e:
            print(f"加载无约束模型权重失败: {e}")
    elif NON_BERT_MODEL_PATH:
        print(f"无约束模型文件 {NON_BERT_MODEL_PATH} 不存在")
    
    # 加载混合模型权重（如果有）
    if os.path.exists(checkpoint_path):
        try:
            print(f"尝试加载混合模型权重从 {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"成功加载混合模型权重从 {checkpoint_path}")
        except Exception as e:
            print(f"加载混合模型权重失败: {e}")
    else:
        print(f"混合模型文件 {checkpoint_path} 不存在")
    
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
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"开始采样过程")
    sample = diffusion.sample(text_prompt)
    print(f"采样完成")
    
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
    print("启动生成脚本")
    # 示例文本提示
    text_prompt = "a house with a red roof"
    print(f"使用文本提示: {text_prompt}")
    generated_voxel = generate_and_save(
        DEFAULT_CHECKPOINT_PATH,
        text_prompt
    )