import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_activation_layer, build_conv_layer, build_norm_layer


class Attention(nn.Module):

    def __init__(self, name, **params):
        super().__init__()

        if name is None:
            self.attention = nn.Identity(**params)
        elif name == 'scse':
            self.attention = SCSEModule(**params)
        else:
            raise ValueError('Attention {} is not implemented'.format(name))

    def forward(self, x):
        return self.attention(x)


class SCSEModule(nn.Module):

    def __init__(self, ch, re=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(ch, ch // re, 1),
            nn.ReLU(inplace=True), nn.Conv2d(ch // re, ch, 1), nn.Sigmoid())
        self.sSE = nn.Sequential(nn.Conv2d(ch, ch, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


class DecoderBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        conv_cfg,
        norm_cfg,
        act_cfg,
        attention_type=None,
    ):
        super().__init__()
        conv1 = [
            build_conv_layer(
                conv_cfg,
                in_channels + skip_channels,
                out_channels,
                kernel_size=3,
                padding=1),
            build_activation_layer(act_cfg)
        ]
        if norm_cfg is not None:
            conv1.append(build_norm_layer(norm_cfg, out_channels)[1])
        self.conv1 = nn.Sequential(*conv1)
        self.attention1 = Attention(
            attention_type, in_channels=in_channels + skip_channels)
        conv2 = [
            build_conv_layer(
                conv_cfg, out_channels, out_channels, kernel_size=3,
                padding=1),
            build_activation_layer(act_cfg)
        ]
        if norm_cfg is not None:
            conv2.append(build_norm_layer(norm_cfg, out_channels)[1])
        self.conv2 = nn.Sequential(*conv2)
        self.attention2 = Attention(attention_type, in_channels=out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class CenterBlock(nn.Sequential):

    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None):
        conv1 = [
            build_conv_layer(
                conv_cfg, in_channels, out_channels, kernel_size=3, padding=1),
            build_activation_layer(act_cfg)
        ]
        if norm_cfg is not None:
            conv1.append(build_norm_layer(norm_cfg, out_channels)[1])
        conv1 = nn.Sequential(*conv1)
        conv2 = [
            build_conv_layer(
                conv_cfg, out_channels, out_channels, kernel_size=3,
                padding=1),
            build_activation_layer(act_cfg)
        ]
        if norm_cfg is not None:
            conv2.append(build_norm_layer(norm_cfg, out_channels)[1])
        conv2 = nn.Sequential(*conv2)
        super().__init__(conv1, conv2)


class SeparableConv2d(nn.Sequential):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
    ):
        dephtwise_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        pointwise_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=bias,
        )
        super().__init__(dephtwise_conv, pointwise_conv)
