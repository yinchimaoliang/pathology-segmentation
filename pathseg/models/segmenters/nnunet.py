import torch.nn as nn

from ..builder import SEGMENTERS, build_decoder, build_encoder


@SEGMENTERS.register_module()
class NNUNet(nn.Module):

    def __init__(self, encoder, decoder, activation='softmax'):
        super().__init__()
        self.encoder = build_encoder(encoder)
        decoder['encoder_channels'] = self.encoder.backbone.out_shapes
        self.decoder = build_decoder(decoder)
        for name, module in self.encoder.backbone.named_children():
            if name == 'relu':
                self.encoder.backbone.add_module('relu',
                                                 nn.LeakyReLU(inplace=True))
            if isinstance(module, nn.Sequential):
                for basic_module in module.modules():
                    for name, layer in basic_module.named_modules():
                        if name == 'relu':
                            basic_module.add_module('relu',
                                                    nn.LeakyReLU(inplace=True))
        for name, module in self.decoder.named_children():
            if name == 'relu':
                self.decoder.backbone.add_module('relu',
                                                 nn.LeakyReLU(inplace=True))
        print(self.encoder)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
