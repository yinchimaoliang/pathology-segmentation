from pathseg.core.utils.dilate_utils import replace_strides_with_dilation
from ..builder import ENCODERS
from .base_encoder import BaseEncoder


@ENCODERS.register_module()
class DeeplabV3PlusEncoder(BaseEncoder):

    def __init__(self, backbone, encoder_output_stride=16):
        super().__init__(backbone)
        if encoder_output_stride == 8:
            self.make_dilated(stage_list=[3, 4], dilation_list=[2, 4])

        elif encoder_output_stride == 16:
            self.make_dilated(stage_list=[4], dilation_list=[2])
        else:
            raise ValueError(
                'Encoder output stride should be 8 or 16, got {}'.format(
                    encoder_output_stride))

    def make_dilated(self, stage_list, dilation_list):
        stages = self.backbone.stages
        for stage_indx, dilation_rate in zip(stage_list, dilation_list):
            replace_strides_with_dilation(
                module=stages[stage_indx],
                dilation_rate=dilation_rate,
            )
