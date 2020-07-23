import torch
import torch.nn as nn

from pathseg.core.base.model import Model
from pathseg.core.common.blocks import SeparableConv2d
from ..builder import DECODERS


@DECODERS.register_module()
class DeepLabV3PlusDecoder(Model):

    def __init__(self,
                 encoder_channels,
                 decoder_channels=[256, 48],
                 output_stride=16,
                 final_channels=2):
        super().__init__()
        if output_stride not in {8, 16}:
            raise ValueError('Output stride should be 8 or 16, got {}.'.format(
                output_stride))

        self.out_channels = decoder_channels
        self.output_stride = output_stride

        scale_factor = 2 if output_stride == 8 else 4
        self.up = nn.UpsamplingBilinear2d(scale_factor=scale_factor)

        self.block1 = nn.Sequential(
            nn.Conv2d(
                encoder_channels[1],
                decoder_channels[1],
                kernel_size=1,
                bias=False),
            nn.BatchNorm2d(decoder_channels[1]),
            nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            SeparableConv2d(
                encoder_channels[0] + decoder_channels[1],
                decoder_channels[0],
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(decoder_channels[0]),
            nn.ReLU(),
        )
        self.final_conv = nn.Conv2d(
            decoder_channels[0], final_channels, kernel_size=(1, 1))

    def forward(self, features):
        aspp_features = self.up(features[0])
        high_res_features = self.block1(features[1])
        concat_features = torch.cat([aspp_features, high_res_features], dim=1)
        fused_features = self.block2(concat_features)
        final_features = self.up(self.final_conv(fused_features))
        return final_features
