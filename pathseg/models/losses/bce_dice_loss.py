import torch.nn as nn

from .bce_loss import BCELoss
from .dice_loss import DiceLoss


class BCEDiceLoss(nn.Module):

    def __init__(self,
                 weight=None,
                 size_average=None,
                 reduce=None,
                 reduction='mean',
                 pos_weight=None,
                 beta=1,
                 eps=1e-7,
                 threshold=None,
                 activation='sigmoid'):
        super().__init__()
        self.bce_loss = BCELoss(weight, size_average, reduce, reduction,
                                pos_weight)
        self.dice_loss = DiceLoss(beta, eps, threshold, activation)

    def forward(self, outputs, annotation):
        return self.bce_loss(outputs, annotation) + self.dice_loss(
            outputs, annotation)
