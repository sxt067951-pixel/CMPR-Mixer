"""Training losses used in the manuscript and the MSE-only ablation."""

import torch
import torch.nn as nn


class FrequencyMSELoss(nn.Module):
    def __init__(self, alpha=.2):
        super().__init__()
        if not 0 <= alpha <= 1:
            raise ValueError("frequency alpha must be in [0, 1]")
        self.alpha = float(alpha)

    def forward(self, prediction, target):
        mse = (prediction - target).square().mean()
        if not self.alpha:
            return mse
        difference = torch.fft.rfft(prediction.float(), dim=1) - torch.fft.rfft(target.float(), dim=1)
        return (1 - self.alpha) * mse + self.alpha * difference.abs().mean()
