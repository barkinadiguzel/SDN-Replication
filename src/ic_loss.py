import torch.nn.functional as F

def ic_loss(pred, target):
    return F.cross_entropy(pred, target)
