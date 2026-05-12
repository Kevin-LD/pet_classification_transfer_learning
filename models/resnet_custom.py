import torch
import torch.nn as nn
from torchvision import models
from .attention import SELayer

# 1. 定义一个包装器，将 SE 加入到 ResNet 的 BasicBlock 之后
class SEBasicBlockWrapper(nn.Module):
    def __init__(self, original_block, channels):
        super(SEBasicBlockWrapper, self).__init__()
        self.original_block = original_block
        self.se = SELayer(channels)

    def forward(self, x):
        x = self.original_block(x) # 先运行原有的 ResNet Block
        x = self.se(x)             # 再运行 SE 层
        return x

def get_baseline_model(model_name='resnet18', num_classes=37, pretrained=True, use_attention=None):
    """
    model_name (str): 'resnet18' 或 'resnet34'
    num_classes (int): 分类数目
    pretrained (bool): 是否使用预训练权重
    use_attention (str): 'SE' 表示加入 SE-block，None 表示原始 ResNet
    """
    
    # 1. 加载模型
    if model_name == 'resnet18':
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
    elif model_name == 'resnet34':
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet34(weights=weights)
    else:
        raise ValueError("Baseline 模型仅支持 'resnet18' 或 'resnet34'")

    # 2. 如果指定了 SE，则手动修改网络结构
    if use_attention == 'SE':
        print(f"Adding SE-blocks to {model_name}...")
        
        # ResNet18/34 由 layer1, layer2, layer3, layer4 组成
        for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            layer = getattr(model, layer_name)
            for i, block in enumerate(layer):
                # 获取该层 block 输出的通道数
                # 在 ResNet 中，卷积层的输出通道数可以从最后一个 bn 层获取
                if isinstance(block, models.resnet.BasicBlock):
                    out_channels = block.bn2.num_features
                    # 用包装了 SE 的新 block 替换原有的 block
                    layer[i] = SEBasicBlockWrapper(block, out_channels)

    # 3. 修改全连接层
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    return model
