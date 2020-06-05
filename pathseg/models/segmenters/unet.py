import torch
import torch.nn as nn

from ..builder import SEGMENTERS, build_decoder, build_encoder


@SEGMENTERS.register_module()
class UNet(nn.Module):

    def __init__(self, encoder, decoder, activation='softmax'):
        super().__init__()
        self.encoder = build_encoder(encoder)
        decoder['encoder_channels'] = self.encoder.backbone.out_shapes
        self.decoder = build_decoder(decoder)
        if activation == 'softmax':
            self.activation = nn.Softmax(dim=1)
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(
                'Activation should be "sigmoid"/"softmax"/callable/None')

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)`
        and apply activation function (if activation is not `None`)
        with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes,
            height, width)

        """
        if self.training:
            self.eval()

        with torch.no_grad():
            x = self.forward(x)
            if self.activation:
                x = self.activation(x[-1])

        self.train()
        return x
