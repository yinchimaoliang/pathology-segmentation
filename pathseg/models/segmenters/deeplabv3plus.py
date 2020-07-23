import torch.nn as nn

from ..builder import SEGMENTERS, build_decoder, build_encoder


@SEGMENTERS.register_module()
class DeeplabV3Plus(nn.Module):

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = build_encoder(encoder)
        self.decoder = build_decoder(decoder)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
