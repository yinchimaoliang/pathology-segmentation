import torch
import torch.nn as nn
from torch.nn import functional as F

from pathseg.core.utils.dilate_utils import replace_strides_with_dilation
from ..builder import ENCODERS
from .base_encoder import BaseEncoder


class ASPPConv(nn.Sequential):

    def __init__(self, in_channels, out_channels, dilation):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )


class ASPPSeparableConv(nn.Sequential):

    def __init__(self, in_channels, out_channels, dilation):
        super().__init__(
            SeparableConv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )


class ASPPPooling(nn.Sequential):

    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(
            x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 atrous_rates,
                 separable=False):
        super(ASPP, self).__init__()
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            ))

        rate1, rate2, rate3 = tuple(atrous_rates)
        ASPPConvModule = ASPPConv if not separable else ASPPSeparableConv

        modules.append(ASPPConvModule(in_channels, out_channels, rate1))
        modules.append(ASPPConvModule(in_channels, out_channels, rate2))
        modules.append(ASPPConvModule(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(
                5 * out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


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


@ENCODERS.register_module()
class DeeplabV3PlusEncoder(BaseEncoder):

    def __init__(self,
                 backbone,
                 encoder_output_stride=16,
                 out_channels=256,
                 atrous_rates=(12, 24, 36)):
        super().__init__(backbone)
        if encoder_output_stride == 8:
            self.make_dilated(stage_list=[3, 4], dilation_list=[2, 4])

        elif encoder_output_stride == 16:
            self.make_dilated(stage_list=[4], dilation_list=[2])
        else:
            raise ValueError(
                'Encoder output stride should be 8 or 16, got {}'.format(
                    encoder_output_stride))

        self.aspp = nn.Sequential(
            ASPP(
                self.backbone.out_shapes[0],
                out_channels,
                atrous_rates,
                separable=True),
            SeparableConv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def make_dilated(self, stage_list, dilation_list):
        stages = self.backbone.stages
        for stage_indx, dilation_rate in zip(stage_list, dilation_list):
            replace_strides_with_dilation(
                module=stages[stage_indx],
                dilation_rate=dilation_rate,
            )

    def forward(self, x):
        features = super().forward(x)
        aspp_features = self.aspp(features[0])
        return [aspp_features, features[3]]
