import os
import torch
from datetime import datetime

# 导入数据加载器
from data_loader import get_dataloader  # 重要！必须添加这一行

# 导入模型和扩散模型
from model import UNet3D
from diffusion import DiffusionModel
from tqdm import tqdm


# 配置参数
DATA_DIR = "blocks"         # HDF5数据目录
SAVE_DIR = "checkpoints"    # 模型保存目录
BATCH_SIZE = 1             # 批量大小（根据显存调整）
RESOLUTION = 32             # 体素分辨率
EPOCHS = 1                # 训练轮数
SAVE_INTERVAL = 1          # 模型保存间隔
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # 自动选择设备

if __name__ == "__main__":
    # 初始化数据加载器
    dataloader = get_dataloader(DATA_DIR, BATCH_SIZE, RESOLUTION)

    # 初始化模型
    model = UNet3D()
    diffusion = DiffusionModel(model, device=DEVICE)

    # 优化器（建议使用AdamW）
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # 学习率调度器（可选）
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 创建保存目录
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 训练循环
    for epoch in range(EPOCHS):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch in progress_bar:
            loss = diffusion.train_step(batch, optimizer)
            total_loss += loss
            progress_bar.set_postfix(loss=f"{loss:.4f}")
        
        # 更新学习率
        scheduler.step()
        
        # 定期保存模型
        if (epoch+1) % SAVE_INTERVAL == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            save_path = os.path.join(SAVE_DIR, f"diffusion_{timestamp}_epoch{epoch+1}.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch+1,
            }, save_path)
            print(f"\n模型已保存至 {save_path}")
            
        print(f"Epoch {epoch+1} 平均损失: {total_loss/len(dataloader):.4f}")
