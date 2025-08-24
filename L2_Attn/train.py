import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from datetime import datetime

# 导入数据加载器
from data_loader import get_dataloader, MinecraftDataset  # 重要！必须添加这一行

# 导入模型和扩散模型
from model import UNet3D
from diffusion import DiffusionModel
from tqdm import tqdm
import torch.multiprocessing as mp

# 添加 DataLoader 导入
from torch.utils.data import DataLoader

# 配置参数
DATA_DIR = "blocks"         # HDF5数据目录
SAVE_DIR = "checkpoints"    # 模型保存目录
BATCH_SIZE = 1             # 批量大小（根据显存调整）
RESOLUTION = 32             # 体素分辨率
EPOCHS = 1                # 训练轮数
SAVE_INTERVAL = 1          # 模型保存间隔

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # 设置环境变量以减少内存碎片
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def main(rank, world_size):
    # 初始化分布式训练环境
    setup(rank, world_size)
    
    # 设置设备
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    
    # 初始化数据集和分布式采样器
    dataset = MinecraftDataset(DATA_DIR, RESOLUTION)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4)
    
    # 初始化模型
    model = UNet3D().to(device)
    model = DDP(model, device_ids=[rank])
    diffusion = DiffusionModel(model, device=device)
    
    # 优化器（建议使用AdamW）
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # 学习率调度器（可选）
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # 创建保存目录（只在主进程中创建）
    if rank == 0:
        os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 训练循环
    for epoch in range(EPOCHS):
        # 设置sampler的epoch以确保正确的shuffle
        sampler.set_epoch(epoch)
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}") if rank == 0 else dataloader
        
        for batch in progress_bar:
            batch = batch.to(device)
            loss = diffusion.train_step(batch, optimizer)
            total_loss += loss
            if rank == 0:
                progress_bar.set_postfix(loss=f"{loss:.4f}")
            
            # 清理缓存以减少内存占用
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # 更新学习率
        scheduler.step()
        
        # 定期保存模型（只在主进程中保存）
        if rank == 0 and (epoch+1) % SAVE_INTERVAL == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            save_path = os.path.join(SAVE_DIR, f"diffusion_{timestamp}_epoch{epoch+1}.pt")
            torch.save({
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch+1,
            }, save_path)
            print(f"\n模型已保存至 {save_path}")
            
        if rank == 0:
            print(f"Epoch {epoch+1} 平均损失: {total_loss/len(dataloader):.4f}")
    
    cleanup()

if __name__ == "__main__":
    # 获取GPU数量
    world_size = torch.cuda.device_count()
    
    # 使用spawn启动多进程训练
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)