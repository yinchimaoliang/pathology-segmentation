import torch.nn as nn

from ..builder import SEGMENTORS, build_decoder, build_encoder


@SEGMENTORS.register_module()
class UNet(nn.Module):

    def __init__(self, encoder, decoder, activation='softmax'):
        super().__init__()
        self.encoder = build_encoder(encoder)
        decoder['encoder_channels'] = self.encoder.backbone.out_shapes
        self.decoder = build_decoder(decoder)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
