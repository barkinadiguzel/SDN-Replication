import torch.nn as nn

def get_activation(name='relu'):
    if name == 'relu':
        return nn.ReLU(inplace=True)
    elif name == 'leaky':
        return nn.LeakyReLU(0.1, inplace=True)
    elif name == 'sigmoid':
        return nn.Sigmoid()
    elif name == 'tanh':
        return nn.Tanh()
    else:
        return nn.Identity()
