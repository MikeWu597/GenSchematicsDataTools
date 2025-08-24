import os
import h5py
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class MinecraftTextDataset(Dataset):
    def __init__(self, data_dir, resolution=32):
        """
        初始化带文本提示的数据集
        :param data_dir: 数据目录，包含HDF5文件和对应的文本描述
        :param resolution: 统一分辨率（32）
        """
        self.data_dir = data_dir
        self.resolution = resolution
        
        # 获取所有HDF5文件
        self.hdf5_files = [f for f in os.listdir(data_dir) if f.endswith('.hdf5')]
        
        # 创建文件名到索引的映射
        self.file_indices = {f.split('.')[0]: i for i, f in enumerate(self.hdf5_files)}
        
    def __len__(self):
        return len(self.hdf5_files)
    
    def __getitem__(self, idx):
        # 获取HDF5文件名
        hdf5_filename = self.hdf5_files[idx]
        base_name = hdf5_filename.split('.')[0]
        
        # 读取体素数据
        with h5py.File(os.path.join(self.data_dir, hdf5_filename), 'r') as f:
            voxel = f['blocks'][:]
        
        # 标准化处理：[0,1] -> [-1,1]
        voxel = voxel.astype(np.float32)
        voxel = (voxel - 0.5) / 0.5  # 二值化数据归一化
        
        # 统一分辨率（如果原始数据不一致）
        if voxel.shape != (self.resolution,)*3:
            voxel = self.resize_voxel(voxel, self.resolution)
            
        # 尝试读取对应的文本描述
        text = ""
        text_file_path = os.path.join(self.data_dir, f"{base_name}.txt")
        if os.path.exists(text_file_path):
            with open(text_file_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
        
        # 如果没有找到文本文件，则使用默认描述
        if not text:
            text = "a minecraft structure"
            
        return {
            'voxel': torch.from_numpy(voxel).unsqueeze(0),  # 添加通道维度 [1,D,H,W]
            'text': text
        }
    
    def resize_voxel(self, voxel, target_size):
        """
        体素插值函数（最近邻插值）
        :param voxel: 原始体素数组 (D,H,W)
        :param target_size: 目标分辨率
        :return: 统一分辨率的体素
        """
        voxel = torch.from_numpy(voxel).unsqueeze(0).unsqueeze(0)  # [1,1,D,H,W]
        resized = torch.nn.functional.interpolate(
            voxel, size=(target_size,)*3, mode='nearest'
        )
        return resized.squeeze().numpy()

def collate_fn(batch):
    """
    自定义批处理函数，将样本组合成批次
    """
    voxels = torch.stack([item['voxel'] for item in batch])
    texts = [item['text'] for item in batch]
    return {
        'voxels': voxels,
        'texts': texts
    }

def get_text_dataloader(data_dir='blocks', batch_size=16, resolution=32):
    """
    创建支持文本的数据加载器
    :param data_dir: 数据目录
    :param batch_size: 批量大小
    :param resolution: 统一分辨率
    :return: DataLoader对象
    """
    dataset = MinecraftTextDataset(data_dir, resolution)
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        collate_fn=collate_fn
    )