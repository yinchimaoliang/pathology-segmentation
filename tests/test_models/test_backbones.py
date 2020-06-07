from pathseg.models import ResNet, build_backbone


def test_resnet():

    cfg = dict(type='ResNet', name='resnet18', weights='imagenet')
    resnet = build_backbone(cfg)
    assert isinstance(resnet, ResNet)
