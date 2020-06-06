from mmcv.utils import Registry, build_from_cfg

EVALS = Registry('eval')


def build_eval(cfg, default_args=None):
    dataset = build_from_cfg(cfg, EVALS, default_args)
    return dataset
