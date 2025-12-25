import torch

def collect_intermediate_features(model, x, layers_idx):
    features = []
    for idx, layer in enumerate(model):
        x = layer(x)
        if idx in layers_idx:
            features.append(x)
    return features
