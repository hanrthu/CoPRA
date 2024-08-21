import torch
import torch.nn as nn


class PearsonCorrLoss(nn.Module):
    def __init__(self):
        super(PearsonCorrLoss, self).__init__()

    def forward(self, pred, target):
        pred_mean = torch.mean(pred)
        target_mean = torch.mean(target)
        
        pred_centered = pred - pred_mean
        target_centered = target - target_mean
        
        numerator = torch.sum(pred_centered * target_centered)
        denominator = torch.sqrt(torch.sum(pred_centered ** 2) * torch.sum(target_centered ** 2))
        
        pearson_corr = numerator / (denominator + 1e-5)
        loss = 1 - pearson_corr
        
        return loss