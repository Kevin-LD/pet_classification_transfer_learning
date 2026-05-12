import torch.nn as nn

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=True),
            nn.Sigmoid()
        )
        self._initialize_weights()

    def _initialize_weights(self):
        # 第一层：标准 Kaiming
        nn.init.kaiming_normal_(self.fc[0].weight, mode='fan_out', nonlinearity='relu')
        
        # 第二层：
        # 为了使用 resnet 预训练权重
        # 我们把权重设为 0，偏置设为一个较大的正数（3.0）
        # 这样 Sigmoid(0 * x + 3.0) ≈ Sigmoid(3.0) ≈ 0.95
        # 模型初始状态接近 1.0 倍缩放，即接近原始 ResNet 的行为
        nn.init.constant_(self.fc[2].weight, 0)
        if self.fc[2].bias is not None:
            nn.init.constant_(self.fc[2].bias, 3.0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
