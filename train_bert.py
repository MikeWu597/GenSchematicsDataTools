import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from datetime import datetime

# 导入数据加载器
from data_loader_bert import get_text_dataloader, MinecraftTextDataset

# 导入模型和扩散模型
from model_bert import UNet3DHybrid
from diffusion_bert import DiffusionModelWithText
from tqdm import tqdm
import torch.multiprocessing as mp

# 添加 DataLoader 导入
from torch.utils.data import DataLoader

# 配置参数
DATA_DIR = "blocks"         # HDF5数据目录
SAVE_DIR = "checkpoints_bert"    # 模型保存目录
BATCH_SIZE = 1             # 批量大小（根据显存调整）
RESOLUTION = 32             # 体素分辨率
EPOCHS = 1                # 训练轮数
SAVE_INTERVAL = 1          # 模型保存间隔
NON_BERT_MODEL_PATH = "checkpoints/diffusion_20250824_1927_epoch200.pt"  # 无约束模型路径

def setup(rank, world_size):
    if torch.cuda.is_available() and world_size > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12356'  # 使用不同的端口避免冲突
        # 设置环境变量以减少内存碎片
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

def main(rank, world_size, use_cuda=True):
    device = torch.device(f"cuda:{rank}" if use_cuda and torch.cuda.is_available() else "cpu")
    print(f"[进程 {rank}] 启动训练进程，使用设备: {device}")
    
    # 初始化分布式训练环境（仅在使用CUDA且有多个GPU时）
    if use_cuda and torch.cuda.is_available() and world_size > 1:
        setup(rank, world_size)
    
    # 初始化数据集
    if not os.path.exists(DATA_DIR):
        print(f"[进程 {rank}] 错误: 数据目录 {DATA_DIR} 不存在")
        if use_cuda and torch.cuda.is_available() and world_size > 1 and dist.is_initialized():
            cleanup()
        return
        
    print(f"[进程 {rank}] 数据目录 {DATA_DIR} 存在")
    
    try:
        dataset = MinecraftTextDataset(DATA_DIR, RESOLUTION)
        print(f"[进程 {rank}] 数据集加载完成，共 {len(dataset)} 个样本")
    except Exception as e:
        print(f"[进程 {rank}] 数据集加载失败: {e}")
        if use_cuda and torch.cuda.is_available() and world_size > 1 and dist.is_initialized():
            cleanup()
        return
    
    # 使用分布式采样器（仅在使用CUDA且有多个GPU时）
    if use_cuda and torch.cuda.is_available() and world_size > 1 and dist.is_initialized():
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        dataloader = DataLoader(
            dataset, 
            batch_size=BATCH_SIZE, 
            sampler=sampler, 
            num_workers=0,  # 减少到0以避免可能的问题
            collate_fn=lambda batch: {
                'voxels': torch.stack([item['voxel'] for item in batch]),
                'texts': [item['text'] for item in batch]
            }
        )
    else:
        # CPU或单GPU训练
        dataloader = DataLoader(
            dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=True,
            num_workers=0,
            collate_fn=lambda batch: {
                'voxels': torch.stack([item['voxel'] for item in batch]),
                'texts': [item['text'] for item in batch]
            }
        )
    
    # 初始化模型
    print(f"[进程 {rank}] 初始化模型")
    model = UNet3DHybrid().to(device)
    print(f"[进程 {rank}] 模型初始化完成")
    
    # 如果指定了无约束模型路径，则加载预训练权重
    if NON_BERT_MODEL_PATH and os.path.exists(NON_BERT_MODEL_PATH):
        try:
            print(f"[进程 {rank}] 尝试加载无约束模型权重从 {NON_BERT_MODEL_PATH}")
            checkpoint = torch.load(NON_BERT_MODEL_PATH, map_location=device)
            # 加载匹配的权重
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if k in model_dict and model_dict[k].shape == v.shape}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            print(f"[进程 {rank}] 成功加载无约束模型权重从 {NON_BERT_MODEL_PATH}")
        except Exception as e:
            print(f"[进程 {rank}] 加载无约束模型权重失败: {e}")
    elif NON_BERT_MODEL_PATH:
        print(f"[进程 {rank}] 无约束模型文件 {NON_BERT_MODEL_PATH} 不存在")
    
    # 冻结BERT相关参数
    frozen_params = []
    for name, param in model.named_parameters():
        if 'text_encoder' in name:
            param.requires_grad = False
            frozen_params.append(name)
    
    if frozen_params:
        print(f"[进程 {rank}] 冻结了 {len(frozen_params)} 个BERT相关参数")
        for name in frozen_params:
            print(f"[进程 {rank}] 冻结参数: {name}")
    else:
        print(f"[进程 {rank}] 没有需要冻结的参数")
    
    # 使用DDP（仅在使用CUDA且有多个GPU且已初始化分布式训练时）
    if use_cuda and torch.cuda.is_available() and world_size > 1 and dist.is_initialized():
        model = DDP(model, device_ids=[rank])
        diffusion = DiffusionModelWithText(model, device=device)
        print(f"[进程 {rank}] 模型已包装为DDP")
    else:
        # CPU或单GPU训练
        diffusion = DiffusionModelWithText(model, device=device)
        print(f"[进程 {rank}] 使用单设备训练模式")
    
    # 优化器（只优化非冻结参数）
    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    print(f"[进程 {rank}] 可训练参数数量: {len(trainable_params)}")
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)
    
    # 学习率调度器（可选）
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # 创建保存目录（只在主进程中创建）
    if rank == 0:
        os.makedirs(SAVE_DIR, exist_ok=True)
        print(f"[进程 {rank}] 模型保存目录: {SAVE_DIR}")
    
    print(f"[进程 {rank}] 开始训练循环，共 {EPOCHS} 轮")
    # 训练循环
    for epoch in range(EPOCHS):
        # 设置sampler的epoch以确保正确的shuffle（仅在使用CUDA且有多个GPU且已初始化分布式训练时）
        if use_cuda and torch.cuda.is_available() and world_size > 1 and dist.is_initialized():
            sampler.set_epoch(epoch)
            
        total_loss = 0
        dataloader_len = len(dataloader)
        print(f"[进程 {rank}] Epoch {epoch+1}/{EPOCHS} 开始，共 {dataloader_len} 个批次")
        
        if dataloader_len == 0:
            print(f"[进程 {rank}] 警告: 数据加载器为空")
            continue
            
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", disable=(rank != 0))
        
        batch_count = 0
        for batch in progress_bar:
            batch_count += 1
            batch['voxels'] = batch['voxels'].to(device)
            loss = diffusion.train_step(batch, optimizer)
            total_loss += loss
            if rank == 0:
                progress_bar.set_postfix(loss=f"{loss:.4f}")
            
            # 清理缓存以减少内存占用
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        print(f"[进程 {rank}] Epoch {epoch+1}/{EPOCHS} 完成，处理了 {batch_count} 个批次")
        
        # 更新学习率
        scheduler.step()
        
        # 定期保存模型（只在主进程中保存）
        if rank == 0 and (epoch+1) % SAVE_INTERVAL == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            save_path = os.path.join(SAVE_DIR, f"diffusion_bert_hybrid_{timestamp}_epoch{epoch+1}.pt")
            state_dict = model.module.state_dict() if use_cuda and torch.cuda.is_available() and world_size > 1 and dist.is_initialized() else model.state_dict()
            torch.save({
                'model_state_dict': state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch+1,
            }, save_path)
            print(f"\n[进程 {rank}] 模型已保存至 {save_path}")
            
        if rank == 0 and dataloader_len > 0:
            print(f"[进程 {rank}] Epoch {epoch+1} 平均损失: {total_loss/dataloader_len:.4f}")
        elif rank == 0:
            print(f"[进程 {rank}] Epoch {epoch+1} 没有处理任何数据")
    
    print(f"[进程 {rank}] 训练完成，清理资源")
    if use_cuda and torch.cuda.is_available() and world_size > 1 and dist.is_initialized():
        cleanup()
    print(f"[进程 {rank}] 资源清理完成")

if __name__ == "__main__":
    print("启动训练脚本")
    # 获取GPU数量
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    print(f"检测到 {world_size} 个计算设备")
    
    use_cuda = torch.cuda.is_available()
    if not use_cuda:
        print("未检测到GPU，将使用CPU进行训练")
    
    # 启动训练
    if use_cuda and world_size > 1:
        print("启动多进程训练")
        mp.spawn(main, args=(world_size, use_cuda), nprocs=world_size, join=True)
    else:
        # CPU或单GPU训练
        main(0, world_size, use_cuda)
    
    print("训练脚本执行完成")