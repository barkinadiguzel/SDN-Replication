import torch.nn as nn
from layers.conv_block import ConvBlock

def make_vgg_block(in_channels, out_channels, num_convs=2):
    layers = []
    for _ in range(num_convs):
        layers.append(ConvBlock(in_channels, out_channels))
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)
