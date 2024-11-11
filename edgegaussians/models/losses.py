import torch.nn as nn
import torch.nn.functional as F
import torch

class MaskedL1Loss(nn.Module):
    def forward(self, input, target, mask):
        return F.l1_loss(input[mask], target[mask])
    
class WeightedL1Loss(nn.Module):
    def forward(self, input, target, weights):
        return torch.mean(weights.to(input.device) * torch.abs(input - target))