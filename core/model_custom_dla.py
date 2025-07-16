import torch
import torch.nn as nn
import torch.nn.functional as F

class AdvancedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.main_path = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.main_path(x) + self.shortcut(x)
        return F.relu(out)


class FeatureAggregator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_list):
        x = torch.cat(x_list, dim=1)
        return self.fuse(x)


class RecursiveFusionBlock(nn.Module):
    def __init__(self, block, in_ch, out_ch, level=1, stride=1):
        super().__init__()
        self.left = block(in_ch, out_ch, stride)
        self.right = block(out_ch, out_ch, 1) if level == 1 else RecursiveFusionBlock(block, out_ch, out_ch, level-1)
        self.merge = FeatureAggregator(out_ch * 2, out_ch)

    def forward(self, x):
        left_out = self.left(x)
        right_out = self.right(left_out)
        return self.merge([left_out, right_out])


class RedefinedDLA(nn.Module):
    def __init__(self, block=AdvancedResidualBlock, num_classes=10):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.stage1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.stage3 = RecursiveFusionBlock(block, 64, 128, level=1)
        self.stage4 = RecursiveFusionBlock(block, 128, 256, level=2, stride=2)
        self.stage5 = RecursiveFusionBlock(block, 256, 512, level=2, stride=2)
        self.stage6 = RecursiveFusionBlock(block, 512, 512, level=1, stride=2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.pre(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.global_pool(x)
        return self.classifier(x)


if __name__ == '__main__':
    model = RedefinedDLA()
    test_input = torch.randn(1, 3, 32, 32)
    test_output = model(test_input)
    print(test_output.shape)  # torch.Size([1, 10])
