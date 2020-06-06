import torch

from pathseg.models.losses import BCEDiceLoss, BCELoss, DiceLoss


def test_bce_loss():
    output = torch.Tensor([[[[0.6, 0.6], [0.6, 0.6]], [[0.4, 0.4], [0.4,
                                                                    0.4]]]])
    annotation = torch.Tensor([[[[1, 1], [1, 1]], [[0, 0], [0, 0]]]])

    bce_loss = BCELoss()
    loss = bce_loss(output, annotation)

    assert torch.isclose(loss, torch.as_tensor(0.6753), 1e-3)


def test_dice_loss():
    output = torch.Tensor([[[[0.6, 0.6], [0.6, 0.6]], [[0.4, 0.4], [0.4,
                                                                    0.4]]]])
    annotation = torch.Tensor([[[[1, 1], [1, 1]], [[0, 0], [0, 0]]]])

    dice_loss = DiceLoss()
    loss = dice_loss(output, annotation)

    assert torch.isclose(loss, torch.as_tensor(0.4246), 1e-3)


def test_bce_dice_loss():
    output = torch.Tensor([[[[0.6, 0.6], [0.6, 0.6]], [[0.4, 0.4], [0.4,
                                                                    0.4]]]])
    annotation = torch.Tensor([[[[1, 1], [1, 1]], [[0, 0], [0, 0]]]])

    bce_dice_loss = BCEDiceLoss()
    loss = bce_dice_loss(output, annotation)

    assert torch.isclose(loss, torch.as_tensor(1.0999), 1e-3)
