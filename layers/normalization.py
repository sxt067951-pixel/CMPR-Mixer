"""Reversible instance normalization, adapted from TimeMixer (Apache-2.0)."""

import torch
import torch.nn as nn


class Normalize(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.affine_weight = nn.Parameter(torch.ones(channels))
        self.affine_bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x, mode):
        if mode == "norm":
            axes = tuple(range(1, x.ndim - 1))
            self.mean = x.mean(dim=axes, keepdim=True).detach()
            self.stdev = torch.sqrt(x.var(dim=axes, keepdim=True, unbiased=False) + self.eps).detach()
            return (x - self.mean) / self.stdev * self.affine_weight + self.affine_bias
        if mode == "denorm":
            return ((x - self.affine_bias) / (self.affine_weight + self.eps ** 2)
                    * self.stdev + self.mean)
        raise ValueError("normalization mode must be norm or denorm")
