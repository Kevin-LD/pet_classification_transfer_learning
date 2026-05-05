import torch
import torch.nn as nn
from torchvision import models

def get_baseline_model(model_name='resnet18', num_classes=37, pretrained=True):
    """
    获取基线模型 (ResNet-18 或 ResNet-34)
    model_name (str): 网络名称，'resnet18' 或 'resnet34'
    num_classes (int): 分类数目，宠物数据集为 37
    pretrained (bool): 是否使用 ImageNet 预训练权重
    """
    
    # 1. 根据参数选择模型架构、导入预训练权重或随机初始化
    if model_name == 'resnet18':
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
    elif model_name == 'resnet34':
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet34(weights=weights)
    else:
        raise ValueError("Baseline 模型仅支持 'resnet18' 或 'resnet34'")

    # 2. 修改全连接层 (输出层)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    return model
