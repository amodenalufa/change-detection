"""
ResNet50-V2 Encoder for Change Detection
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ResNet50Encoder(nn.Module):
    """
    ResNet50-V2 encoder
    Outputs 4-scale features: [1/4, 1/8, 1/16, 1/32] with channels [256, 512, 1024, 2048]
    """

    def __init__(self, pretrained=True):
        super(ResNet50Encoder, self).__init__()

        try:
            # torchvision >= 0.13, PyTorch >= 1.12
            from torchvision.models import ResNet50_Weights
            if pretrained:
                # V1
                weights = ResNet50_Weights.IMAGENET1K_V1  # 或 IMAGENET1K_V2 如果有
                resnet = models.resnet50(weights=weights)
                print("[Encoder] Using ResNet50-IMAGENET1K_V1 (new API)")
            else:
                resnet = models.resnet50(weights=None)
        except ImportError:
            # torchvision < 0.13, PyTorch 1.11
            resnet = models.resnet50(pretrained=pretrained)
            if pretrained:
                print("[Encoder] Using ResNet50-ImageNet (legacy API)")

        # 各层输出通道数
        self.channels = [256, 512, 1024, 2048]

        # 提取各组件
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1  # 1/4, 256 channels
        self.layer2 = resnet.layer2  # 1/8, 512 channels
        self.layer3 = resnet.layer3  # 1/16, 1024 channels
        self.layer4 = resnet.layer4  # 1/32, 2048 channels

    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W]
        Returns:
            list: [feat_1/4, feat_1/8, feat_1/16, feat_1/32]
        """
        features = []

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        features.append(x)  # [B, 256, H/4, W/4]

        x = self.layer2(x)
        features.append(x)  # [B, 512, H/8, W/8]

        x = self.layer3(x)
        features.append(x)  # [B, 1024, H/16, W/16]

        x = self.layer4(x)
        features.append(x)  # [B, 2048, H/32, W/32]

        return features

    def get_channels(self):
        return self.channels


class ChangeDetectionEncoder(nn.Module):
    """
    Dual-temporal shared encoder using ResNet50-V2
    """

    def __init__(self, pretrained=True):
        super(ChangeDetectionEncoder, self).__init__()

        self.encoder = ResNet50Encoder(pretrained=pretrained)
        self.channels = self.encoder.channels

        if pretrained:
            print("[Encoder] ResNet50 loaded with shared weights")

    def forward(self, img_t1, img_t2):
        """
        Args:
            img_t1: [B, 3, H, W] - time 1 image
            img_t2: [B, 3, H, W] - time 2 image
        Returns:
            feats_t1, feats_t2: lists of 4-scale features
        """
        feats_t1 = self.encoder(img_t1)
        feats_t2 = self.encoder(img_t2)
        return feats_t1, feats_t2

    def get_channel_list(self):
        return self.channels