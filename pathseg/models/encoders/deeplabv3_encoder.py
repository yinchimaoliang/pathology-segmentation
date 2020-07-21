import segmentation_models_pytorch as smp

from ..builder import ENCODERS
from .base_encoder import BaseEncoder

model = smp.DeepLabV3Plus(encoder_weights=None)
print(model)


@ENCODERS.register_module()
class DeeplabV3PlusEncoder(BaseEncoder):

    def __init__(self, backbone):
        super().__init__(backbone)
