import torch
import torch.nn as nn

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16, pretrained_init=True):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=True),
            nn.Sigmoid()
        )
        # 将初始化参数传入
        self._initialize_weights(pretrained_init)

    def _initialize_weights(self, pretrained_init):
        if pretrained_init:
            # 适配预训练权重的初始化
            # 第一层：标准 Kaiming
            nn.init.kaiming_normal_(self.fc[0].weight, mode='fan_out', nonlinearity='relu')
            if self.fc[0].bias is not None:
                nn.init.constant_(self.fc[0].bias, 0)
            
            # 第二层：初始化为接近 0，偏置为较大正数
            # 使 Sigmoid(0*x + 3) ≈ 0.95，接近 1.0 倍缩放
            nn.init.constant_(self.fc[2].weight, 0)
            if self.fc[2].bias is not None:
                nn.init.constant_(self.fc[2].bias, 3.0)
        else:
            # Kaiming 初始化
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
