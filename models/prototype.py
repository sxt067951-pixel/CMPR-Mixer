"""Fixed-capacity dot-product retrieval; no prototype growth or alignment."""

import torch
import torch.nn as nn


class PrototypeLayer(nn.Module):
    """Learnable keys and temporal values with load-conditioned amplitudes."""

    def __init__(self, prototype_dim, input_length, output_length, count=8, exog_dim=0):
        super().__init__()
        self.prototype_dim = prototype_dim
        self.input_length = input_length
        self.output_length = output_length
        self.output_dim = 1
        self.prototype_num = [count]
        self.layer_num = 1
        self.exog_dim = exog_dim
        self.layer_bias = nn.ModuleList([nn.Sequential(
            nn.Linear(prototype_dim, prototype_dim * 2, bias=False), nn.ReLU(),
            nn.Linear(prototype_dim * 2, count, bias=False))])
        self.prototype_embedding = nn.ParameterList([nn.Parameter(torch.zeros(count, prototype_dim))])
        self.prototype_value = nn.ParameterList([nn.Parameter(torch.zeros(count, output_length, 1))])
        self.patch_embedding = nn.Sequential(
            nn.Linear(input_length, input_length * 2), nn.ReLU(),
            nn.Linear(input_length * 2, prototype_dim))
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.kaiming_uniform_(self.prototype_embedding[0], nonlinearity="relu")
        if exog_dim:
            if exog_dim != 56:
                raise ValueError("context retrieval expects the documented 56-column exogenous schema")
            self.continuous_embedding = nn.Sequential(
                nn.Linear(36, prototype_dim), nn.GELU(), nn.Linear(prototype_dim, prototype_dim))
            self.hour_embedding = nn.Embedding(24, prototype_dim)
            self.weekday_embedding = nn.Embedding(7, prototype_dim)
            self.holiday_embedding = nn.Embedding(8, prototype_dim)
            self.extreme_type_embedding = nn.Embedding(12, prototype_dim)
            self.context_norm = nn.LayerNorm(prototype_dim)
            self.context_temporal_pool = nn.Linear(input_length, 1, bias=False)
            nn.init.constant_(self.context_temporal_pool.weight, 1.0 / input_length)
            self.query_fusion_gate = nn.Linear(prototype_dim * 2, prototype_dim)
            nn.init.zeros_(self.query_fusion_gate.weight)
            nn.init.constant_(self.query_fusion_gate.bias, 1.38629436112)
            for module in self.continuous_embedding.modules():
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_uniform_(module.weight, mode="fan_out", nonlinearity="relu")
                    nn.init.zeros_(module.bias)

    @staticmethod
    def _one_hot_group_id(group):
        active = group > 0
        category = active.float().argmax(dim=-1) + 1
        return torch.where(active.any(dim=-1), category, torch.zeros_like(category)).long()

    @staticmethod
    def _time_index(normalized_value, max_index):
        return torch.round((normalized_value + 0.5) * max_index).long().clamp(0, max_index)

    def _context_query(self, context):
        external, marks = context[:, :, :56], context[:, :, 56:]
        hour = self._time_index(marks[:, :, 0], 23)
        weekday = self._time_index(marks[:, :, 1], 6)
        holiday = self._one_hot_group_id(external[:, :, 37:44])
        event = self._one_hot_group_id(external[:, :, 45:56])
        values = (self.continuous_embedding(external[:, :, :36])
                  + self.hour_embedding(hour) + self.weekday_embedding(weekday)
                  + self.holiday_embedding(holiday) + self.extreme_type_embedding(event))
        return self.context_temporal_pool(self.context_norm(values).transpose(1, 2)).transpose(1, 2)

    def forward(self, x, context=None):
        batch = x.size(0)
        load_query = self.patch_embedding(x.mean(dim=-1, keepdim=True).transpose(1, 2))
        query = load_query
        if self.exog_dim and context is not None:
            context_query = self._context_query(context)
            gate = torch.sigmoid(self.query_fusion_gate(torch.cat([load_query, context_query], dim=-1)))
            query = gate * load_query + (1 - gate) * context_query
        scores = torch.einsum("...ij,...jk->...ik", query,
            self.prototype_embedding[0].unsqueeze(0).expand(batch, -1, -1).transpose(1, 2)).softmax(dim=-1)
        amplitudes = self.layer_bias[0](load_query).abs().reshape(batch, self.prototype_num[0], 1, 1)
        values = amplitudes * self.prototype_value[0].unsqueeze(0)
        residual = (scores.reshape(batch, -1, 1, 1) * values).sum(dim=1)
        return [residual], [scores]
