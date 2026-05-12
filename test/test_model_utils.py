import os
import sys
import torch
import torch.nn as nn

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.resnet_custom import get_baseline_model
from utils.optim import set_parameter_requires_grad, get_optimizer

def test_model_structure():
    print(">>> 测试 1: 模型结构与输出层修改")
    num_classes = 37
    model = get_baseline_model(model_name='resnet18', num_classes=num_classes, pretrained=False)
    
    # 检查输出维度
    out_features = model.fc.out_features
    print(f"模型类型: ResNet-18")
    print(f"预期输出维度: {num_classes}, 实际输出维度: {out_features}")
    assert out_features == num_classes, "输出维度错误！"
    return model

def test_freeze_logic(model):
    print(">>> 测试 2: 冻结逻辑 (set_parameter_requires_grad)")
    
    # 场景 A: 冻结 Backbone
    set_parameter_requires_grad(model, freeze_backbone=True)
    backbone_frozen = not model.conv1.weight.requires_grad
    head_trainable = model.fc.weight.requires_grad
    print(f"冻结 Backbone 模式 -> Backbone 冻结: {backbone_frozen}, Head 可训练: {head_trainable}")
    assert backbone_frozen and head_trainable, "冻结逻辑 A 出错！"
    
    # 场景 B: 全部解冻
    set_parameter_requires_grad(model, freeze_backbone=False)
    backbone_trainable = model.conv1.weight.requires_grad
    print(f"全解冻模式 -> Backbone 可训练: {backbone_trainable}, Head 可训练: {head_trainable}")
    assert backbone_trainable, "冻结逻辑 B 出错！"

def test_optimizer_groups(model):
    print(">>> 测试 3: 优化器参数分组与 Weight Decay")
    
    # 设置特定超参数
    lr_backbone = 1e-5
    lr_head = 1e-3
    wd = 1e-4
    
    # 确保模型是全解冻状态以进行分组测试
    set_parameter_requires_grad(model, freeze_backbone=False)
    
    optimizer = get_optimizer(
        model, 
        opt_type='AdamW', 
        lr_backbone=lr_backbone, 
        lr_head=lr_head, 
        weight_decay=wd
    )
    
    print(f"参数组总数: {len(optimizer.param_groups)} (预期应为 4)")
    
    # 验证每一个组的属性
    group_counts = {"backbone_decay": 0, "backbone_no_decay": 0, "head_decay": 0, "head_no_decay": 0}
    
    for i, group in enumerate(optimizer.param_groups):
        lr = group['lr']
        curr_wd = group['weight_decay']
        num_params = len(group['params'])
        
        # 通过学习率和衰减率判定它是哪个组
        if lr == lr_backbone and curr_wd == wd:
            tag = "Backbone (Decay)"
        elif lr == lr_backbone and curr_wd == 0.0:
            tag = "Backbone (No-Decay)"
        elif lr == lr_head and curr_wd == wd:
            tag = "Head (Decay)"
        elif lr == lr_head and curr_wd == 0.0:
            tag = "Head (No-Decay)"
        else:
            tag = "Unknown"

        print(f"  Group {i}: {tag} | LR: {lr:.1e} | WD: {curr_wd:.1e} | 参数数量: {num_params}")

    # 简单的 sanity check: 验证没有 WD 的参数是否包含 Bias 或 BN 权重
    # 我们检查 Head (No-Decay) 是否只有 1 个参数（即 model.fc.bias）
    for group in optimizer.param_groups:
        if group['lr'] == lr_head and group['weight_decay'] == 0.0:
            # Linear 层的 bias 是 1D，应在此组
            has_bias = any(p.ndimension() == 1 for p in group['params'])
            print(f"  验证: Head (No-Decay) 包含 Bias 项: {has_bias}")


if __name__ == "__main__":
    # 执行测试流
    baseline_model = test_model_structure()
    test_freeze_logic(baseline_model)
    test_optimizer_groups(baseline_model)
