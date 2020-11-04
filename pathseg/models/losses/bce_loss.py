import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES


@LOSSES.register_module()
class BCELoss(nn.Module):

    def __init__(self,
                 weight=None,
                 size_average=None,
                 reduce=None,
                 reduction='mean',
                 pos_weight=None):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(weight, size_average, reduce,
                                             reduction, pos_weight)

    def forward(self, outputs, annotation):
        loss = 0
        if type(outputs).__name__ == 'list':
            for output in outputs:
                output = F.interpolate(
                    output, size=[annotation.shape[2], annotation.shape[3]])
                loss += self.bce_loss(output, annotation)
        else:
            loss += self.bce_loss(outputs, annotation)

        return loss
