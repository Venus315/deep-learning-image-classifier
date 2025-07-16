import torch
import torch.nn as nn
import torch.nn.functional as F

MODERN_VGG_LAYOUT = {
    'LiteVGG': [64, 'BN', 'P', 128, 'BN', 'P', 256, 256, 'BN', 'P', 512, 'BN', 'P'],
    'DeepVGG': [64, 64, 'BN', 'P', 128, 128, 'BN', 'P', 256, 256, 256, 'BN', 'P', 512, 512, 512, 'BN', 'P']
}

class EnhancedVGG(nn.Module):
    def __init__(self, layout_key='LiteVGG', num_classes=10, in_channels=3):
        super().__init__()
        self.features = self._build_layers(MODERN_VGG_LAYOUT[layout_key], in_channels)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

    def _build_layers(self, config, in_channels):
        layers = []
        for v in config:
            if v == 'P':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            elif v == 'BN':
                layers.append(nn.BatchNorm2d(in_channels))
            else:
                layers.append(nn.Conv2d(in_channels, v, kernel_size=3, padding=1))
                layers.append(nn.ReLU(inplace=True))
                in_channels = v
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        return nn.Sequential(*layers)

# 快速测试
if __name__ == '__main__':
    model = EnhancedVGG(layout_key='LiteVGG')
    dummy_input = torch.randn(1, 3, 32, 32)
    output = model(dummy_input)
    print(output.shape)  # 形状应为 [1, 10]