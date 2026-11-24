"""Hourly value/time embedding, adapted from TimeMixer (Apache-2.0)."""

import torch.nn as nn


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super().__init__()
        self.tokenConv = nn.Conv1d(c_in, d_model, kernel_size=3, padding=1,
                                   padding_mode="circular", bias=False)
        nn.init.kaiming_normal_(self.tokenConv.weight, mode="fan_in",
                                nonlinearity="leaky_relu")

    def forward(self, x):
        return self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.embed = nn.Linear(4, d_model, bias=False)

    def forward(self, marks):
        return self.embed(marks)


class DataEmbedding(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.value_embedding = TokenEmbedding(1, d_model)
        self.temporal_embedding = TimeFeatureEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values, marks):
        if values is None:
            return self.temporal_embedding(marks)
        embedded = self.value_embedding(values)
        if marks is not None:
            embedded = embedded + self.temporal_embedding(marks)
        return self.dropout(embedded)
