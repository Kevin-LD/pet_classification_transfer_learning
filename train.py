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
from models.resnet_custom import get_baseline_model
from utils.optim import get_optimizer, get_scheduler
from utils.eval import evaluate


def train_one_epoch(model, dataloader, criterion, optimizer, device, save_history=True):
    """
    训练一个 Epoch 的逻辑
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    iter_stats = []

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

            if save_history:
                iter_stats.append({'loss': loss.item()})

            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total:.4f}'})

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, iter_stats


def run_training(args):
    """
    主训练流程控制
    """
    # 自动选择硬件
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    # 1. 断点续训 (Resume) 预处理
    start_epoch = 0
    best_val_acc = 0.0
    history = {'epoch_history': [], 'iter_history': []}
    checkpoint = None

    if args.resume and os.path.isfile(args.resume):
        # 如果是续训，默认保存路径设为权重所在的文件夹
        checkpoint_dir = os.path.dirname(os.path.abspath(args.resume))
        args.save_dir = checkpoint_dir
        print(f"检测到续训模式，自动同步保存目录至: {args.save_dir}")

        print(f"正在加载 Checkpoint: '{args.resume}'")
        checkpoint = torch.load(args.resume, map_location=device)
        
        # 恢复模型配置参数
        preserved_args = ['resume', 'epochs', 'save_dir', 'save_history', 'warmup_epochs']
        checkpoint_args = checkpoint.get('args', {})
        for k, v in checkpoint_args.items():
            if k not in preserved_args:
                setattr(args, k, v)
        
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        
        # 检查已训练的 Epoch 数是否超过当前参数
        if start_epoch >= args.epochs:
            print("\n" + "!"*60)
            print(f"警告: 提供的 Checkpoint 已经训练了 {start_epoch} 个 Epoch。")
            print(f"当前的 --epochs 参数设置为 {args.epochs}，无需继续训练。")
            print(f"建议: 请将 --epochs 设置为大于 {start_epoch} 的数值。")
            print("!"*60 + "\n")
            return  # 退出训练

        if args.save_history:
            old_history = checkpoint.get('history', [])
            history = old_history if isinstance(old_history, dict) else {'epoch_history': old_history, 'iter_history': []}
            
        print(f"成功恢复配置。将从 Epoch {start_epoch+1} 续训至 Epoch {args.epochs}。")

    # 2. 确定保存目录与 Config 写入
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    
    config_path = os.path.join(save_dir, 'config.json')
    if not args.resume or not os.path.exists(config_path):
        with open(config_path, 'w') as f:
            json.dump(vars(args), f, indent=4)

    # 打印配置
    print("\n" + "-"*30)
    print("最终实验参数配置:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    print("-" * 30 + "\n")
    print(f"Using hardware device: {device}")

    # 3. 准备数据
    train_loader, val_loader = get_train_val_dataloaders(
        batch_size=args.batch_size, image_size=224, num_workers=args.num_workers
    )

    # 4. 初始化
    model = get_baseline_model(
        model_name=args.model_name, 
        num_classes=37, 
        pretrained=args.pretrained,
        use_attention=args.use_attention
    ).to(device)

    optimizer = get_optimizer(model, args.optimizer, args.lr_backbone, args.lr_head, args.weight_decay)
    scheduler = get_scheduler(optimizer, args)
    criterion = nn.CrossEntropyLoss()

    if checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # 为 optimizer.param_groups 打上 is_head 标签
    head_param_ids = {id(p) for n, p in model.named_parameters() if n.startswith("fc.")}
    for group in optimizer.param_groups:
        # 如果组内有任何一个参数属于 head，则标记此组为 head 组
        group['is_head'] = any(id(p) in head_param_ids for p in group['params'])

    # 5. 训练循环
    start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        print(f"\n[Epoch {epoch+1}/{args.epochs}]")
        
        is_warmup = epoch < args.warmup_epochs

        # 动态冻结/解冻 Backbone & LR 覆盖 
        # 修复：同样使用 startswith 精确控制 head
        for name, param in model.named_parameters():
            if not name.startswith("fc."):
                param.requires_grad = not is_warmup

        # 对 LR 进行备份和覆写
        for group in optimizer.param_groups:
            if not group['is_head'] and is_warmup:
                group['backup_lr'] = group['lr']  # 备份真实调度的 LR
                group['lr'] = 0.0                 # 强制设为 0
        
        # 记录当前 Epoch 实际生效的 LR
        epoch_head_lr = next((g['lr'] for g in optimizer.param_groups if g['is_head']), 0.0)
        epoch_backbone_lr = next((g['lr'] for g in optimizer.param_groups if not g['is_head']), 0.0)

        # 运行训练
        train_loss, train_acc, iter_stats = train_one_epoch(model, train_loader, criterion, optimizer, device, args.save_history)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # Scheduler Step 前恢复真实 LR 
        for group in optimizer.param_groups:
            if 'backup_lr' in group:
                group['lr'] = group['backup_lr']
                del group['backup_lr']

        # 更新 Scheduler
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_acc)
            else:
                scheduler.step()

        print(f"Train | Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"Val   | Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | Head LR: {epoch_head_lr:.6e} | Backbone LR: {epoch_backbone_lr:.6e}")

        # 6. 记录与保存
        if args.save_history:
            history['epoch_history'].append({
                'epoch': epoch + 1, 
                'train_loss': train_loss, 
                'train_acc': train_acc,
                'val_loss': val_loss, 
                'val_acc': val_acc, 
                'head_lr': epoch_head_lr,
                'backbone_lr': epoch_backbone_lr
            })
            history['iter_history'].extend(iter_stats)
            with open(os.path.join(save_dir, 'history.json'), 'w') as f:
                json.dump(history, f, indent=4)

        checkpoint_dict = {
            'epoch': epoch,
            'model_name': args.model_name,
            'use_attention': args.use_attention,
            'best_val_acc': max(val_acc, best_val_acc),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'history': history if args.save_history else None,
            'args': vars(args)
        }

        torch.save(checkpoint_dict, os.path.join(save_dir, "latest_model.pth"))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(checkpoint_dict, os.path.join(save_dir, "best_model.pth"))
            print(f"发现最佳模型! Val Acc: {best_val_acc:.4f}")

    total_time = time.time() - start_time
    print(f"\n训练完成! 总耗时: {total_time/60:.2f} 分钟. Best Val Acc: {best_val_acc:.4f}")

    return best_val_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Oxford-IIIT Pet 图像分类训练脚本")

    # 动态生成默认保存路径
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    default_save_dir = os.path.join('./runs', f'exp_{current_time}')

    # --- 目录与功能开关 ---
    parser.add_argument('--save_dir', type=str, default=default_save_dir)
    parser.add_argument('--no_save_history', dest='save_history', action='store_false')
    parser.set_defaults(save_history=True)
    parser.add_argument('--resume', type=str, default='', help='续训权重路径 (.pth)')

    # --- 训练参数 ---
    parser.add_argument('--model_name', type=str, default='resnet34', choices=['resnet18', 'resnet34'])
    parser.add_argument('--pretrained', dest='pretrained', action='store_true')
    parser.add_argument('--no_pretrained', dest='pretrained', action='store_false')
    parser.set_defaults(pretrained=True)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--warmup_epochs', type=int, default=0, help='冻结 Backbone 进行 Warmup 的 Epoch 数量')
    parser.add_argument('--use_attention', type=str, default=None, choices=['SE'], help='是否在残差块中加入注意力机制')

    # --- 优化器与 Scheduler ---
    parser.add_argument('--optimizer', type=str, default='AdamW', choices=['Adam', 'AdamW', 'SGD'])
    parser.add_argument('--lr_backbone', type=float, default=1e-5)
    parser.add_argument('--lr_head', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--lr_scheduler', type=str, default='CosineAnnealingLR')
    parser.add_argument('--lr_step_size', type=int, default=7)
    parser.add_argument('--lr_gamma', type=float, default=0.1)

    args = parser.parse_args()
    run_training(args)
