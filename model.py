import torch
import torch.nn as nn
import torch.nn.functional as F


class DSConv(nn.Module):

    def __init__(self, in_c, out_c, stride=1):
        super().__init__()

        self.depth = nn.Conv2d(
            in_c,
            in_c,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_c,
            bias=False
        )

        self.point = nn.Conv2d(
            in_c,
            out_c,
            kernel_size=1,
            bias=False
        )

        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        x = self.depth(x)
        x = self.point(x)
        x = self.bn(x)

        return self.relu(x)


class EfficientSegNet(nn.Module):

    def __init__(self, num_classes=21):
        super().__init__()

        self.enc1 = DSConv(3, 20, stride=2)
        self.enc2 = DSConv(20, 40, stride=2)
        self.enc3 = DSConv(40, 64, stride=2)

        self.bottleneck = DSConv(64, 64)

        self.dec3 = DSConv(64, 40)
        self.dec2 = DSConv(40, 20)
        self.dec1 = DSConv(20, 20)

        self.head = nn.Conv2d(20, num_classes, 1)

    def forward(self, x):

        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)

        x = self.bottleneck(x)

        x = F.interpolate(x, scale_factor=2)
        x = self.dec3(x)

        x = F.interpolate(x, scale_factor=2)
        x = self.dec2(x)

        x = F.interpolate(x, scale_factor=2)
        x = self.dec1(x)

        x = self.head(x)

        return x