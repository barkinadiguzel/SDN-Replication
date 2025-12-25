import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """A basic convolutional block: Conv2d -> Normalization -> Activation"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, norm='batch', act='relu'):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        
        if norm == 'batch':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'layer':
            self.norm = nn.LayerNorm([out_channels, 1, 1])
        else:
            self.norm = nn.Identity()
        
        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act == 'leaky':
            self.act = nn.LeakyReLU(0.1, inplace=True)
        else:
            self.act = nn.Identity()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x
