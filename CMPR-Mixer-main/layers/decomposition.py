"""Moving-average decomposition, adapted from TimeMixer (Apache-2.0)."""

import torch
import torch.nn as nn


class MovingAverage(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 != 1:
            raise ValueError("moving_avg must be a positive odd number")
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size, stride=1)

    def forward(self, x):
        padding = (self.kernel_size - 1) // 2
        front = x[:, :1, :].repeat(1, padding, 1)
        end = x[:, -1:, :].repeat(1, padding, 1)
        return self.avg(torch.cat([front, x, end], dim=1).transpose(1, 2)).transpose(1, 2)


class SeriesDecomposition(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = MovingAverage(kernel_size)

    def forward(self, x):
        trend = self.moving_avg(x)
        return x - trend, trend
