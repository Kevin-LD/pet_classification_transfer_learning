import os
import sys
import time
import json
import argparse
from datetime import datetime
import torch
import torch.nn as nn
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from utils.dataset import get_train_val_dataloaders
from models.baseline import get_baseline_model
from utils.optim import get_optimizer, get_scheduler
from utils.eval import evaluate

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    训练一个 Epoch 的逻辑
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # 使用 with 语句，确保每轮结束后释放进度条资源
    with tqdm(dataloader, desc="Training", leave=False) as pbar:
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total:.4f}'})

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def run_training(args):
    """
    主训练流程控制
    """
    # 打印当前的训练配置
    print("\n" + "-"*30)
    print("实验参数配置:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    print("-" * 30 + "\n")

    # 自动选择 GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using hardware device: {device}")

    # 1. 确定保存目录
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    print(f"实验数据将保存至: {save_dir}")

    # 保存训练参数配置作为元数据
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    # 2. 准备数据
    print("加载数据中 (Train & Val)...")
    train_loader, val_loader = get_train_val_dataloaders(
        batch_size=args.batch_size, 
        image_size=224, 
        num_workers=args.num_workers
    )

    # 3. 初始化模型、优化器、调度器与损失函数
    print(f"初始化模型: {args.model_name} (Pretrained: {args.pretrained})")
    model = get_baseline_model(
        model_name=args.model_name, 
        num_classes=37, 
        pretrained=args.pretrained
    )
    model = model.to(device)

    optimizer = get_optimizer(
        model, 
        opt_type=args.optimizer, 
        lr_backbone=args.lr_backbone, 
        lr_head=args.lr_head, 
        weight_decay=args.weight_decay
    )
    
    # 获取学习率调度器
    scheduler = get_scheduler(optimizer, args)
    
    criterion = nn.CrossEntropyLoss()

    # 4. 断点续训 (Resume) 逻辑
    start_epoch = 0
    best_val_acc = 0.0
    history = []

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"正在加载 Checkpoint: '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device)
            
            # 恢复状态
            start_epoch = checkpoint['epoch'] + 1
            best_val_acc = checkpoint.get('best_val_acc', 0.0)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # 恢复 Scheduler 状态
            if scheduler and checkpoint.get('scheduler_state_dict'):
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            if args.save_history:
                history = checkpoint.get('history', [])
            
            print(f"成功从 Epoch {start_epoch} 恢复训练 (历史最佳 Acc: {best_val_acc:.4f})")
        else:
            print(f"警告: 未找到 Checkpoint: '{args.resume}'，将从头开始训练。")

    # 5. 开始 Epoch 循环
    print("\n" + "="*40)
    print("开始训练")
    print("="*40)
    
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        print(f"\n[Epoch {epoch+1}/{args.epochs}]")
        
        # 训练阶段
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        # 验证阶段
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # Scheduler 更新逻辑
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_acc)
            else:
                scheduler.step()

        # 获取当前 Epoch 的学习率 (取 backbone 的 lr)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Train | Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"Val   | Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | LR: {current_lr:.6e}")

        # 记录历史
        if args.save_history:
            history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'lr': current_lr
            })
            # 每个 epoch 结束都输出历史
            history_path = os.path.join(save_dir, 'history.json')
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=4)

        # 准备要保存的 Checkpoint 字典
        checkpoint_dict = {
            'epoch': epoch,
            'model_name': args.model_name,
            'pretrained': args.pretrained,
            'num_classes': 37,
            'best_val_acc': max(val_acc, best_val_acc),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'history': history,
            'args': vars(args)
        }

        # 随时保存最新模型以供意外中断恢复
        latest_save_path = os.path.join(save_dir, "latest_model.pth")
        torch.save(checkpoint_dict, latest_save_path)

        # 保存最佳模型 (基于验证集表现)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_save_path = os.path.join(save_dir, "best_model.pth")
            torch.save(checkpoint_dict, best_save_path)
            print(f"发现最佳模型! Val Acc: {best_val_acc:.4f}. 已保存至: {best_save_path}")

    total_time = time.time() - start_time
    print("\n" + "="*40)
    print(f"训练完成! 总耗时: {total_time/60:.2f} 分钟.")
    print(f"最高验证准确率 (Best Val Acc): {best_val_acc:.4f}")
    if args.save_history:
        print(f"训练历史记录已保存至: {os.path.join(save_dir, 'history.json')}")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Oxford-IIIT Pet 图像分类训练脚本")

    # 动态生成默认保存路径
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    default_save_dir = os.path.join('./runs', f'exp_{current_time}')

    # --- 目录与功能开关 ---
    parser.add_argument('--save_dir', type=str, default=default_save_dir, help='保存实验结果和权重的目录')
    parser.add_argument('--no_save_history', dest='save_history', action='store_false', help='关闭保存训练历史记录 (JSON)')
    parser.set_defaults(save_history=True)
    parser.add_argument('--resume', type=str, default='', help='填写 .pth 文件的路径以断点续训')

    # --- 模型相关参数 ---
    parser.add_argument('--model_name', type=str, default='resnet18', choices=['resnet18', 'resnet34'], help='选择使用的网络模型')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='使用 ImageNet 预训练权重')
    parser.add_argument('--no_pretrained', dest='pretrained', action='store_false', help='从零开始随机初始化')
    parser.set_defaults(pretrained=True)

    # --- 训练循环超参数 ---
    parser.add_argument('--epochs', type=int, default=15, help='训练总轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批大小')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader 线程数')

    # --- 优化器与调度器超参数 ---
    parser.add_argument('--optimizer', type=str, default='AdamW', choices=['Adam', 'AdamW', 'SGD'], help='优化器类型')
    parser.add_argument('--lr_backbone', type=float, default=1e-5, help='主干网络的学习率')
    parser.add_argument('--lr_head', type=float, default=1e-3, help='分类头的学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减系数 (L2 正则化)')
    
    # --- LR Scheduler 参数 ---
    parser.add_argument('--lr_scheduler', type=str, default='CosineAnnealingLR', choices=['None', 'StepLR', 'CosineAnnealingLR', 'ReduceLROnPlateau'], help='学习率调度器类型')
    parser.add_argument('--lr_step_size', type=int, default=7, help='StepLR 的步长')
    parser.add_argument('--lr_gamma', type=float, default=0.1, help='StepLR 学习率衰减率')

    args = parser.parse_args()
    
    run_training(args)
