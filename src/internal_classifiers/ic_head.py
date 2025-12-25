import torch
import torch.nn as nn
import torch.nn.functional as F

class ICHead(nn.Module):
    def __init__(self, in_channels, num_classes, pool_size=4):
        super().__init__()
        self.pool_size = pool_size
        self.feature_reduce = nn.AdaptiveAvgPool2d((pool_size, pool_size))  
        self.fc = nn.Linear(in_channels * pool_size * pool_size, num_classes)
    
    def forward(self, x):
        x = self.feature_reduce(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
