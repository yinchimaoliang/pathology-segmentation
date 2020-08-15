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
            if name.startswith('bn'):
                self.encoder.backbone.add_module(
                    name, nn.InstanceNorm2d(num_features=module.num_features))
            if name == 'relu':
                self.encoder.backbone.add_module('relu',
                                                 nn.LeakyReLU(inplace=True))
            if isinstance(module, nn.Sequential):
                for basic_module in module.modules():
                    for name, layer in basic_module.named_modules():
                        if name == 'relu':
                            basic_module.add_module('relu',
                                                    nn.LeakyReLU(inplace=True))
                        if name.startswith('bn'):
                            basic_module.add_module(
                                name,
                                nn.InstanceNorm2d(
                                    num_features=layer.num_features))
        for decoder_block_name, decoder_block in self.decoder.named_children():
            if decoder_block_name == 'final_conv':
                continue
            for name, conv2d_relu_module in decoder_block.block.named_children(
            ):
                conv2d_relu_module.block.add_module('2',
                                                    nn.LeakyReLU(inplace=True))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
