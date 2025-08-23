import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class MinecraftDataset(Dataset):
    def __init__(self, data_dir, resolution=32):
        """
        初始化数据集
        :param data_dir: HDF5文件所在目录（如'blocks'）
        :param resolution: 统一分辨率（32）
        """
        self.data_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.hdf5')]
        self.resolution = resolution
        
    def __len__(self):
        # 返回数据集大小（99个文件）
        return len(self.data_paths)
    
    def __getitem__(self, idx):
        # 读取单个HDF5文件
        with h5py.File(self.data_paths[idx], 'r') as f:
            voxel = f['blocks'][:]
        
        # 标准化处理：[0,1] -> [-1,1]
        voxel = voxel.astype(np.float32)
        voxel = (voxel - 0.5) / 0.5  # 二值化数据归一化
        
        # 统一分辨率（如果原始数据不一致）
        if voxel.shape != (self.resolution,)*3:
            voxel = self.resize_voxel(voxel, self.resolution)
            
        return torch.from_numpy(voxel).unsqueeze(0)  # 添加通道维度 [1,D,H,W]
    
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

def get_dataloader(data_dir='blocks', batch_size=16, resolution=32):
    """
    创建数据加载器
    :param data_dir: 数据目录
    :param batch_size: 批量大小
    :param resolution: 统一分辨率
    :return: DataLoader对象
    """
    dataset = MinecraftDataset(data_dir, resolution)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
