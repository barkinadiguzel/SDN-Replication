import torch.nn.functional as F

def final_loss(pred, target):
    return F.cross_entropy(pred, target)
