import torch
import torch.nn as nn
import timm

class VisionTransformerTiny(nn.Module):
    """
    ViT-Tiny 模型定义
    """
    def __init__(self, num_classes=37, pretrained=True):
        super(VisionTransformerTiny, self).__init__()
        
        # 使用 timm 创建 vit_tiny_patch16_224 模型
        # pretrained=True 会自动下载并在 ImageNet 上预训练的权重
        self.model = timm.create_model('vit_tiny_patch16_224', pretrained=pretrained)
        
        # 修改最后的分类头 (Head)
        # ViT-Tiny 的 head 包含一个 LayerNorm 和一个 Linear 层
        n_features = self.model.head.in_features
        self.model.head = nn.Linear(n_features, num_classes)
        

    def forward(self, x):
        return self.model(x)

def get_vit_tiny(num_classes=37, pretrained=True):
    """
    方便在 train.py 中调用
    """
    return VisionTransformerTiny(num_classes=num_classes, pretrained=pretrained)
