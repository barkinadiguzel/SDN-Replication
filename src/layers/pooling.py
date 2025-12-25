import torch
import torch.nn as nn

class MixedPooling(nn.Module):
    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size, stride)
        self.avgpool = nn.AvgPool2d(kernel_size, stride)
        self.alpha = nn.Parameter(torch.tensor(0.5)) 
    
    def forward(self, x):
        return self.alpha * self.maxpool(x) + (1 - self.alpha) * self.avgpool(x)
