import torch
import torch.optim.lr_scheduler as lr_scheduler

def set_parameter_requires_grad(model, freeze_backbone=True):
    """
    控制是否冻结主干网络。
    """
    if freeze_backbone:
        for name, param in model.named_parameters():
            # 只有全连接层不冻结
            if not name.startswith("fc."):
                param.requires_grad = False
    else:
        for param in model.parameters():
            param.requires_grad = True

def get_optimizer(model, opt_type='AdamW', lr_backbone=1e-5, lr_head=1e-3, weight_decay=1e-4):
    """
    优化器构建函数
    """
    
    # 准备四个参数组的容器
    params_groups = {
        "backbone_decay": {"params": [], "lr": lr_backbone, "weight_decay": weight_decay},
        "backbone_no_decay": {"params": [], "lr": lr_backbone, "weight_decay": 0.0},
        "head_decay": {"params": [], "lr": lr_head, "weight_decay": weight_decay},
        "head_no_decay": {"params": [], "lr": lr_head, "weight_decay": 0.0},
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        is_head = name.startswith("fc.")
        
        # 只要是 1D 参数（Bias, BN 权重/偏置），就不进行 Weight Decay
        if param.ndimension() == 1:
            group_key = "head_no_decay" if is_head else "backbone_no_decay"
        else:
            group_key = "head_decay" if is_head else "backbone_decay"
            
        params_groups[group_key]["params"].append(param)

    # 过滤空组并构建参数列表
    final_params = [v for v in params_groups.values() if len(v["params"]) > 0]

    # 实例化优化器
    if opt_type == 'AdamW':
        optimizer = torch.optim.AdamW(final_params)
    elif opt_type == 'Adam':
        optimizer = torch.optim.Adam(final_params)
    elif opt_type == 'SGD':
        # SGD 通常配合 momentum 使用
        optimizer = torch.optim.SGD(final_params, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer type: {opt_type}")
        
    return optimizer

def get_scheduler(optimizer, args):
    """
    根据参数获取学习率调度器
    optimizer: 优化器实例
    args: 包含调度器配置的参数对象
    """
    sched_type = getattr(args, 'lr_scheduler', 'StepLR')
    
    if sched_type == 'StepLR':
        # 每隔 step_size 个 epoch，学习率乘以 gamma
        return lr_scheduler.StepLR(
            optimizer, 
            step_size=getattr(args, 'lr_step_size', 7), 
            gamma=getattr(args, 'lr_gamma', 0.1)
        )
        
    elif sched_type == 'CosineAnnealingLR':
        # 余弦退火调度，学习率按余弦曲线下降
        return lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=getattr(args, 'epochs', 15),
            eta_min=1e-6
        )
        
    elif sched_type == 'ReduceLROnPlateau':
        # 当验证集指标停止改善时降低学习率
        return lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max',      # 监控的是准确率，所以是 max
            factor=0.1, 
            patience=3
        )
    
    else:
        return None
