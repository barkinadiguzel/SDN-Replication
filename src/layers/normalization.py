import torch.nn as nn

def get_normalization(name='batch', num_channels=None):
    if name == 'batch':
        return nn.BatchNorm2d(num_channels)
    elif name == 'layer':
        return nn.LayerNorm([num_channels, 1, 1])
    else:
        return nn.Identity()
