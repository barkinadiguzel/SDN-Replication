import torch
import torch.nn as nn
from backbone.vgg_blocks import make_vgg_block
from internal_classifiers.ic_head import ICHead

class SDNNet(nn.Module):
    def __init__(self, num_classes=10, ic_positions=None):
        super().__init__()
        self.num_classes = num_classes
        
        self.features = nn.ModuleList([
            make_vgg_block(3, 64),
            make_vgg_block(64, 128),
            make_vgg_block(128, 256),
            make_vgg_block(256, 512),
            make_vgg_block(512, 512)
        ])
        
        self.ic_positions = ic_positions or [0, 1, 2, 3, 4]
        self.ic_heads = nn.ModuleList([
            ICHead(out_channels, num_classes) 
            for out_channels in [64, 128, 256, 512, 512]
        ])
        
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.final_fc = nn.Linear(512 * 4 * 4, num_classes)

    def forward(self, x):
        features = []
        ic_outputs = []
        
        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx in self.ic_positions:
                ic_idx = self.ic_positions.index(idx)
                ic_out = self.ic_heads[ic_idx](x)
                ic_outputs.append(ic_out)
        
        out = self.avgpool(x)
        out = torch.flatten(out, 1)
        final_out = self.final_fc(out)
        
        return ic_outputs, final_out
