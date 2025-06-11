import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel, stride, kernel // 2, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class C2fBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=1):
        super().__init__()
        self.blocks = nn.Sequential(
            *[ConvBlock(in_channels, out_channels) for _ in range(num_blocks)]
        )

    def forward(self, x):
        return self.blocks(x)


class SPPFBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool_size=5):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, kernel=1, stride=1)
        self.pool = nn.MaxPool2d(pool_size, stride=1, padding=pool_size // 2)
        self.conv2 = ConvBlock(out_channels * 4, out_channels, kernel=1, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.conv2(torch.cat([x, y1, y2, y3], 1))


# Helper for concatenation
class Concat(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, xs):
        return torch.cat(xs, self.dim)


# Simple Detect head for demonstration (not a full YOLO head)
class Detect(nn.Module):
    def __init__(self, num_classes=12, ch=[64, 128, 256]):
        super().__init__()
        self.detect0 = nn.Conv2d(ch[0], (num_classes + 5) * 3, 1)
        self.detect1 = nn.Conv2d(ch[1], (num_classes + 5) * 3, 1)
        self.detect2 = nn.Conv2d(ch[2], (num_classes + 5) * 3, 1)

    def forward(self, feats):
        return [self.detect0(feats[0]), self.detect1(feats[1]), self.detect2(feats[2])]


class YOLOv8nCustom(nn.Module):
    def __init__(self, num_classes=12):
        super().__init__()
        # Backbone
        self.layer0 = ConvBlock(3, 16, 3, 2)  # 0
        self.layer1 = ConvBlock(16, 32, 3, 2)  # 1
        self.layer2 = C2fBlock(32, 32, 1)  # 2
        self.layer3 = ConvBlock(32, 64, 3, 2)  # 3
        self.layer4 = C2fBlock(64, 64, 2)  # 4
        self.layer5 = ConvBlock(64, 128, 3, 2)  # 5
        self.layer6 = C2fBlock(128, 128, 2)  # 6
        self.layer7 = ConvBlock(128, 256, 3, 2)  # 7
        self.layer8 = C2fBlock(256, 256, 1)  # 8
        self.layer9 = SPPFBlock(256, 256, 5)  # 9

        # Head
        self.head_conv10 = ConvBlock(256, 128, kernel=1, stride=1)  # 10
        self.upsample11 = nn.Upsample(scale_factor=2, mode="nearest")  # 11
        self.concat12 = Concat(1)  # 12
        self.c2f13 = C2fBlock(256, 128, 1)  # 13

        self.head_conv14 = ConvBlock(128, 64, kernel=1, stride=1)  # 14
        self.upsample15 = nn.Upsample(scale_factor=2, mode="nearest")  # 15
        self.concat16 = Concat(1)  # 16
        self.c2f17 = C2fBlock(128, 64, 1)  # 17

        self.head_conv18 = ConvBlock(64, 64, kernel=3, stride=2)  # 18
        self.concat19 = Concat(1)  # 19
        self.c2f20 = C2fBlock(128, 128, 1)  # 20

        self.head_conv21 = ConvBlock(128, 128, kernel=3, stride=2)  # 21
        self.concat22 = Concat(1)  # 22
        self.c2f23 = C2fBlock(256, 256, 1)  # 23

        self.detect = Detect(num_classes=num_classes, ch=[64, 128, 256])  # 24

    def forward(self, x):
        # Backbone
        x0 = self.layer0(x)  # 0
        x1 = self.layer1(x0)  # 1
        x2 = self.layer2(x1)  # 2
        x3 = self.layer3(x2)  # 3
        x4 = self.layer4(x3)  # 4
        x5 = self.layer5(x4)  # 5
        x6 = self.layer6(x5)  # 6
        x7 = self.layer7(x6)  # 7
        x8 = self.layer8(x7)  # 8
        x9 = self.layer9(x8)  # 9

        # Head
        x10 = self.head_conv10(x9)  # 10
        x11 = self.upsample11(x10)  # 11
        x12 = self.concat12([x11, x6])  # 12, [11, 6]
        x13 = self.c2f13(x12)  # 13

        x14 = self.head_conv14(x13)  # 14
        x15 = self.upsample15(x14)  # 15
        x16 = self.concat16([x15, x4])  # 16, [15, 4]
        x17 = self.c2f17(x16)  # 17

        x18 = self.head_conv18(x17)  # 18
        x19 = self.concat19([x18, x14])  # 19, [18, 14]
        x20 = self.c2f20(x19)  # 20

        x21 = self.head_conv21(x20)  # 21
        x22 = self.concat22([x21, x9])  # 22, [21, 9]
        x23 = self.c2f23(x22)  # 23

        # Detect on [x17, x20, x23]
        out = self.detect([x17, x20, x23])
        return out


# Example usage:
if __name__ == "__main__":
    model = YOLOv8nCustom(num_classes=12)
    dummy = torch.randn(1, 3, 640, 640)
    outs = model(dummy)
    for i, o in enumerate(outs):
        print(f"Output {i} shape:", o.shape)  # Output shape: [1, (12+5)*3, H, W]
