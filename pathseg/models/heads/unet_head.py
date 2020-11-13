import torch.nn as nn

from pathseg.core.common.blocks import CenterBlock, DecoderBlock
from ..builder import HEADS
from .decode_head import BaseDecodeHead


@HEADS.register_module()
class UNetHead(BaseDecodeHead):

    def __init__(self, num_blocks=5, center=False, **kwargs):
        super(UNetHead, self).__init__(**kwargs)

        assert len(
            self.in_channels
        ) == num_blocks, f'Model depth is {num_blocks}, but you provide ' \
                         f'`decoder_channels` for ' \
                         f'{len(self.in_channels)} blocks.'
        encoder_channels = self.in_channels[::-1]
        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        decoder_channels = self.channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        if center:
            self.center = CenterBlock(head_channels, head_channels,
                                      self.conv_cfg, self.norm_cfg,
                                      self.act_cfg)
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, self.conv_cfg, self.norm_cfg,
                         self.act_cfg) for in_ch, skip_ch, out_ch in zip(
                             in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def compute_channels(self, encoder_channels, decoder_channels):
        channels = [
            encoder_channels[0] + encoder_channels[1],
            encoder_channels[2] + decoder_channels[0],
            encoder_channels[3] + decoder_channels[1],
            encoder_channels[4] + decoder_channels[2],
            0 + decoder_channels[3],
        ]
        return channels

    def forward(self, features):

        # reverse channels to start from head of encoder
        features = features[::-1]

        head = features[0]
        skips = features[1:]

        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        return x
