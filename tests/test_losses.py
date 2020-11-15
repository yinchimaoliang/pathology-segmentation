import pytest
import torch

from pathseg.models import BCEDiceLoss, BCELoss, DiceLoss, build_loss


def test_bce_loss():
    output = torch.Tensor([[[[0.6, 0.6], [0.6, 0.6]], [[0.4, 0.4], [0.4,
                                                                    0.4]]]])
    annotation = torch.Tensor([[[[1, 1], [1, 1]], [[0, 0], [0, 0]]]])

    cfg = dict(type='BCELoss')
    bce_loss = build_loss(cfg)
    assert isinstance(bce_loss, BCELoss)
    loss = bce_loss(output, annotation)

    assert torch.isclose(loss, torch.as_tensor(0.6753), 1e-3)


def test_dice_loss():
    output = torch.Tensor([[[[0.6, 0.6], [0.6, 0.6]], [[0.4, 0.4], [0.4,
                                                                    0.4]]]])
    annotation = torch.Tensor([[[[1, 1], [1, 1]], [[0, 0], [0, 0]]]])

    cfg = dict(type='DiceLoss')
    dice_loss = build_loss(cfg)
    assert isinstance(dice_loss, DiceLoss)
    loss = dice_loss(output, annotation)

    assert torch.isclose(loss, torch.as_tensor(0.4246), 1e-3)


def test_bce_dice_loss():
    output = torch.Tensor([[[[0.6, 0.6], [0.6, 0.6]], [[0.4, 0.4], [0.4,
                                                                    0.4]]]])
    annotation = torch.Tensor([[[[1, 1], [1, 1]], [[0, 0], [0, 0]]]])
    cfg = dict(type='BCEDiceLoss')
    bce_dice_loss = build_loss(cfg)
    assert isinstance(bce_dice_loss, BCEDiceLoss)
    loss = bce_dice_loss(output, annotation)

    assert torch.isclose(loss, torch.as_tensor(1.0999), 1e-3)


def test_ce_loss():
    from pathseg.models import build_loss

    # use_mask and use_sigmoid cannot be true at the same time
    with pytest.raises(AssertionError):
        loss_cfg = dict(
            type='CrossEntropyLoss',
            use_mask=True,
            use_sigmoid=True,
            loss_weight=1.0)
        build_loss(loss_cfg)

    # test loss with class weights
    loss_cls_cfg = dict(
        type='CrossEntropyLoss',
        use_sigmoid=False,
        class_weight=[0.8, 0.2],
        loss_weight=1.0)
    loss_cls = build_loss(loss_cls_cfg)
    fake_pred = torch.Tensor([[100, -100]])
    fake_label = torch.Tensor([1]).long()
    assert torch.allclose(loss_cls(fake_pred, fake_label), torch.tensor(40.))

    loss_cls_cfg = dict(
        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
    loss_cls = build_loss(loss_cls_cfg)
    assert torch.allclose(loss_cls(fake_pred, fake_label), torch.tensor(200.))

    loss_cls_cfg = dict(
        type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)
    loss_cls = build_loss(loss_cls_cfg)
    assert torch.allclose(loss_cls(fake_pred, fake_label), torch.tensor(0.))
