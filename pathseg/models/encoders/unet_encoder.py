from ..builder import ENCODERS
from .base_encoder import BaseEncoder


@ENCODERS.register_module()
class UnetEncoder(BaseEncoder):

    def __init__(self, backbone):
        super().__init__(backbone)
