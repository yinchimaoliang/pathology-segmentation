import torch.nn as nn


def replace_strides_with_dilation(module, dilation_rate):
    """Patch Conv2d modules replacing strides with dilation"""
    for mod in module.modules():
        if isinstance(mod, nn.Conv2d):
            mod.stride = (1, 1)
            mod.dilation = (dilation_rate, dilation_rate)
            kh, kw = mod.kernel_size
            mod.padding = ((kh // 2) * dilation_rate,
                           (kh // 2) * dilation_rate)

            # Kostyl for EfficientNet
            if hasattr(mod, 'static_padding'):
                mod.static_padding = nn.Identity()
