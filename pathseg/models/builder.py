from mmcv.utils import Registry, build_from_cfg
from torch import nn

BACKBONES = Registry('backbone')
ENCODERS = Registry('encoder')
DECODERS = Registry('decoder')
SEGMENTORS = Registry('segmentor')
LOSSES = Registry('loss')
HEADS = Registry('head')


def build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_backbone(cfg):
    return build(cfg, BACKBONES)


def build_encoder(cfg):
    return build(cfg, ENCODERS)


def build_decoder(cfg):
    return build(cfg, DECODERS)


def build_head(cfg):
    return build(cfg, HEADS)


def build_segmentor(cfg, train_cfg=None, test_cfg=None):
    """Build segmentor."""
    return build(cfg, SEGMENTORS, dict(train_cfg=train_cfg, test_cfg=test_cfg))


def build_loss(cfg):
    return build(cfg, LOSSES)
