from mmcv.utils import Registry, build_from_cfg
from torch import nn

BACKBONES = Registry('backbone')
ENCODERS = Registry('encoder')


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
