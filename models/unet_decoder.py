"""
UNet Decoder for Change Detection
Progressive upsampling with skip connections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        B, C, H, W = x.shape
        half = C // 2
        t1 = x[:, :half]
        t2 = x[:, half:]
        
        diff = torch.abs(t1 - t2).mean(dim=1, keepdim=True)
        attn = self.conv(x)
        
        weighted = x * (attn + 0.5)
        return weighted


class ConvBlock(nn.Module):
    """Basic conv block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU"""
    
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.conv(x)


class UpConv(nn.Module):
    """Upsampling conv module"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super(UpConv, self).__init__()

        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False)
            self.bn = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if hasattr(self, 'bn'):
            x = self.up(x)
            x = self.bn(x)
            x = self.relu(x)
        else:
            x = self.up(x)
        return x


class UNetDecoder(nn.Module):
    """
    UNet decoder with progressive upsampling
    """

    def __init__(self, encoder_channels, num_classes=2, bilinear=True, use_skip_attention=True):
        super(UNetDecoder, self).__init__()
        
        self.encoder_channels = encoder_channels
        self.use_skip_attention = use_skip_attention
        decoder_channels = [256, 128, 64, 32]

        # Center block
        self.center = ConvBlock(encoder_channels[-1], encoder_channels[0])

        # Decoder layers
        self.upconv4 = UpConv(encoder_channels[0], decoder_channels[0], bilinear)
        self.conv4 = ConvBlock(encoder_channels[-2] + decoder_channels[0], decoder_channels[0])

        self.upconv3 = UpConv(decoder_channels[0], decoder_channels[1], bilinear)
        self.conv3 = ConvBlock(encoder_channels[-3] + decoder_channels[1], decoder_channels[1])

        self.upconv2 = UpConv(decoder_channels[1], decoder_channels[2], bilinear)
        self.conv2 = ConvBlock(encoder_channels[-4] + decoder_channels[2], decoder_channels[2])

        self.upconv1 = UpConv(decoder_channels[2], decoder_channels[3], bilinear)
        self.conv1 = ConvBlock(decoder_channels[3], decoder_channels[3])

        self.upconv0 = UpConv(decoder_channels[3], decoder_channels[3], bilinear)
        self.conv0 = ConvBlock(decoder_channels[3], decoder_channels[3])
        
        if use_skip_attention:
            self.skip_attns = nn.ModuleList([
                SkipAttention(encoder_channels[-2]),  # 1/16
                SkipAttention(encoder_channels[-3]),  # 1/8
                SkipAttention(encoder_channels[-4]),  # 1/4
            ])

        # Final output
        self.final_conv = nn.Conv2d(decoder_channels[3], num_classes, 1, bias=True)
        
        nn.init.kaiming_normal_(self.final_conv.weight, mode='fan_out', nonlinearity='linear')
        if self.final_conv.bias is not None:
            nn.init.constant_(self.final_conv.bias, 0.0)

    def forward(self, features):
        """
        Args:
            features: list of fused features [f_1/4, f_1/8, f_1/16, f_1/32]
        """
        x = self.center(features[-1])

        # 1/32 -> 1/16
        x = self.upconv4(x)
        skip = features[-2]
        if self.use_skip_attention:
            skip = self.skip_attns[0](skip)
        x = torch.cat([x, skip], dim=1)
        x = self.conv4(x)

        # 1/16 -> 1/8
        x = self.upconv3(x)
        skip = features[-3]
        if self.use_skip_attention:
            skip = self.skip_attns[1](skip)
        x = torch.cat([x, skip], dim=1)
        x = self.conv3(x)

        # 1/8 -> 1/4
        x = self.upconv2(x)
        skip = features[-4]
        if self.use_skip_attention:
            skip = self.skip_attns[2](skip)
        x = torch.cat([x, skip], dim=1)
        x = self.conv2(x)

        # 1/4 -> 1/2 (no skip)
        x = self.upconv1(x)
        x = self.conv1(x)

        # 1/2 -> 1/1 (no skip)
        x = self.upconv0(x)
        x = self.conv0(x)

        out = self.final_conv(x)
        return out


class ChangeDetectionDecoder(nn.Module):
    """
    Change detection decoder with feature fusion
    """

    def __init__(self, encoder_channels, num_classes=2, bilinear=True, fusion='concat'):
        """
        Args:
            encoder_channels: encoder channel dimensions
            num_classes: number of output classes
            bilinear: use bilinear upsampling
            fusion: feature fusion method ('diff', 'concat', 'sum')
        """
        super(ChangeDetectionDecoder, self).__init__()

        self.fusion = fusion

        if fusion == 'concat':
            fusion_channels = [c * 2 for c in encoder_channels]
        else:
            fusion_channels = encoder_channels

        self.decoder = UNetDecoder(fusion_channels, num_classes, bilinear)

    def forward(self, feats_t1, feats_t2):
        """
        Args:
            feats_t1: time 1 feature list
            feats_t2: time 2 feature list
        Returns:
            out: change detection result [B, num_classes, H, W]
        """
        if self.fusion == 'diff':
            fused = [torch.abs(f1 - f2) for f1, f2 in zip(feats_t1, feats_t2)]
        elif self.fusion == 'concat':
            fused = [torch.cat([f1, f2], dim=1) for f1, f2 in zip(feats_t1, feats_t2)]
        elif self.fusion == 'sum':
            fused = [f1 + f2 for f1, f2 in zip(feats_t1, feats_t2)]
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion}")

        return self.decoder(fused)