"""
Baseline Change Detection Model
ResNet50-V2 Encoder + UNet Decoder
"""

import torch
import torch.nn as nn
from .resnet_encoder import ChangeDetectionEncoder
from .unet_decoder import ChangeDetectionDecoder


class ChangeDetectionModel(nn.Module):
    """
    Baseline change detection model
    Dual-branch ResNet50 encoder + UNet decoder
    """

    def __init__(self, num_classes=2, pretrained=True, bilinear=True, fusion='diff'):
        """
        Args:
            num_classes: number of output classes (default 2 for change/no-change)
            pretrained: use ImageNet pretrained weights (recommended)
            bilinear: use bilinear upsampling (True) or transposed conv (False)
            fusion: feature fusion method ('diff', 'concat', 'sum')
        """
        super(ChangeDetectionModel, self).__init__()

        self.encoder = ChangeDetectionEncoder(pretrained=pretrained)
        encoder_channels = self.encoder.channels

        self.decoder = ChangeDetectionDecoder(
            encoder_channels=encoder_channels,
            num_classes=num_classes,
            bilinear=bilinear,
            fusion=fusion
        )

    def forward(self, img_t1, img_t2):
        """
        Args:
            img_t1: time 1 image [B, 3, H, W]
            img_t2: time 2 image [B, 3, H, W]
        Returns:
            out: change detection result [B, num_classes, H, W]
        """
        feats_t1, feats_t2 = self.encoder(img_t1, img_t2)
        out = self.decoder(feats_t1, feats_t2)
        return out

    def predict(self, img_t1, img_t2):
        """
        Prediction interface
        Args:
            img_t1: time 1 image [B, 3, H, W]
            img_t2: time 2 image [B, 3, H, W]
        Returns:
            change_prob: change probability map [B, H, W]
            change_mask: binary change mask [B, H, W]
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(img_t1, img_t2)
            probs = torch.softmax(logits, dim=1)
            change_prob = probs[:, 1, :, :]
            change_mask = torch.argmax(logits, dim=1)
        return change_prob, change_mask


def create_model(config):
    """
    Create model from config
    Args:
        config: configuration dictionary
    Returns:
        model: ChangeDetectionModel instance
    """
    num_classes = config.get('num_classes', 2)
    pretrained = config.get('pretrained', True)
    bilinear = config.get('bilinear', True)
    fusion = config.get('fusion', 'diff')

    if 'backbone' in config and config['backbone'] != 'resnet50':
        print(f"[Warning] backbone='{config['backbone']}' ignored. Using ResNet50.")

    model = ChangeDetectionModel(
        num_classes=num_classes,
        pretrained=pretrained,
        bilinear=bilinear,
        fusion=fusion
    )

    return model