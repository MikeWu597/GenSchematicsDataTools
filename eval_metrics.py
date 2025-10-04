import torch
import torch.nn.functional as F
import numpy as np

def chamfer_distance(pc1, pc2):
    """
    计算两个点云之间的Chamfer Distance
    
    Args:
        pc1: 第一个点云，形状为 [B, N, 3] 或 [N, 3]
        pc2: 第二个点云，形状为 [B, M, 3] 或 [M, 3]
    
    Returns:
        chamfer_dist: 两个点云之间的Chamfer Distance
    """
    # 如果输入是numpy数组，转换为torch张量
    if isinstance(pc1, np.ndarray):
        pc1 = torch.from_numpy(pc1)
    if isinstance(pc2, np.ndarray):
        pc2 = torch.from_numpy(pc2)
    
    # 确保输入在正确的设备上
    if pc1.is_cuda != pc2.is_cuda:
        if pc1.is_cuda:
            pc2 = pc2.to(pc1.device)
        else:
            pc1 = pc1.to(pc2.device)
    
    # 如果是单个点云（没有batch维度），添加batch维度
    if pc1.dim() == 2:
        pc1 = pc1.unsqueeze(0)
    if pc2.dim() == 2:
        pc2 = pc2.unsqueeze(0)
    
    # 计算点云之间的距离矩阵
    # pc1: [B, N, 3], pc2: [B, M, 3]
    # dist_matrix: [B, N, M]
    dist_matrix = torch.cdist(pc1, pc2)
    
    # 计算从pc1到pc2的最小距离
    min_dist_pc1_to_pc2 = torch.min(dist_matrix, dim=2)[0]  # [B, N]
    
    # 计算从pc2到pc1的最小距离
    min_dist_pc2_to_pc1 = torch.min(dist_matrix, dim=1)[0]  # [B, M]
    
    # 计算Chamfer Distance
    chamfer_dist_pc1_to_pc2 = torch.mean(min_dist_pc1_to_pc2, dim=1)  # [B]
    chamfer_dist_pc2_to_pc1 = torch.mean(min_dist_pc2_to_pc1, dim=1)  # [B]
    
    # 双向Chamfer Distance
    chamfer_dist = chamfer_dist_pc1_to_pc2 + chamfer_dist_pc2_to_pc1  # [B]
    
    return chamfer_dist

def voxel_to_point_cloud(voxel, threshold=0.5):
    """
    将体素网格转换为点云
    
    Args:
        voxel: 体素网格，形状为 [D, H, W] 或 [1, D, H, W] 或 [B, 1, D, H, W]
        threshold: 阈值，用于确定哪些体素被激活
    
    Returns:
        point_cloud: 点云，形状为 [N, 3]
    """
    # 如果输入是torch张量，转换为numpy数组
    if torch.is_tensor(voxel):
        voxel = voxel.detach().cpu().numpy()
    
    # 确保输入是正确的形状
    while len(voxel.shape) > 3:
        voxel = voxel.squeeze(0)
    
    # 找到激活的体素
    if voxel.ndim == 3:
        points = np.argwhere(voxel > threshold)
    else:
        raise ValueError("体素网格必须是3D的")
    
    # 如果没有激活的体素，返回空点云
    if len(points) == 0:
        return np.empty((0, 3))
    
    # 转换为点云
    point_cloud = points.astype(np.float32)
    
    return point_cloud