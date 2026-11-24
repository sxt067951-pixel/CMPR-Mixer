"""CMPR-Mixer: context-guided prototype retrieval and multiscale mixing.

Forecasting backbone adapted from TimeMixer under Apache-2.0. Only the
channel-independent, hourly load-forecasting path used by this project remains.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.decomposition import SeriesDecomposition
from layers.embedding import DataEmbedding
from layers.normalization import Normalize
from models.prototype import PrototypeLayer


class AnchorPooling(nn.Module):
    """Per-channel mean + tanh(beta) * sigmoid(gate) * (max - mean)."""

    def __init__(self, channels, stride):
        super().__init__()
        self.stride = stride
        self.mean_weight = nn.Parameter(torch.zeros(1, channels, 1))
        self.max_weight = nn.Parameter(torch.zeros(1, channels, 1))
        self.gate_bias = nn.Parameter(torch.zeros(1, channels, 1))
        self.residual_scale = nn.Parameter(torch.zeros(1, channels, 1))
        self.register_buffer("_gate_sum", torch.zeros(()), persistent=False)
        self.register_buffer("_effective_abs_sum", torch.zeros(()), persistent=False)
        self.register_buffer("_stat_count", torch.zeros(()), persistent=False)

    def reset_statistics(self):
        self._gate_sum.zero_()
        self._effective_abs_sum.zero_()
        self._stat_count.zero_()

    def get_statistics(self):
        if self._stat_count.item() == 0:
            return float("nan"), float("nan")
        return ((self._gate_sum / self._stat_count).item(),
                (self._effective_abs_sum / self._stat_count).item())

    def forward(self, x):
        mean = F.avg_pool1d(x, self.stride, self.stride)
        peak = F.max_pool1d(x, self.stride, self.stride)
        gate = torch.sigmoid(self.mean_weight * mean + self.max_weight * peak + self.gate_bias)
        strength = torch.tanh(self.residual_scale) * gate
        if not self.training:
            with torch.no_grad():
                self._gate_sum.add_(gate.detach().sum())
                self._effective_abs_sum.add_(strength.detach().abs().sum())
                self._stat_count.add_(gate.numel())
        return mean + strength * (peak - mean)


class ExogenousFeatureGate(nn.Module):
    def __init__(self, exog_dim):
        super().__init__()
        self.exog_dim = exog_dim
        self.gate_logits = nn.Parameter(torch.zeros(exog_dim))

    def forward(self, x):
        external = x[:, :, :self.exog_dim] * torch.sigmoid(self.gate_logits).view(1, 1, -1)
        return torch.cat([external, x[:, :, self.exog_dim:]], dim=-1)

    def get_weights(self):
        weights = torch.sigmoid(self.gate_logits.detach()).cpu()
        return {"weight": weights.tolist(), "mean_abs_weight": weights.abs().mean().item()}


class MultiScaleMixing(nn.Module):
    def __init__(self, configs, reverse=False):
        super().__init__()
        self.reverse = reverse
        self.use_exog_prototype = bool(configs.use_exog_prototype and configs.load_exog_dim)
        indices = list(range(configs.down_sampling_layers))
        if reverse:
            indices.reverse()
        self.prototype_bank = nn.ModuleList()
        for index in indices:
            fine = configs.seq_len // configs.down_sampling_window ** index
            coarse = fine // configs.down_sampling_window
            source, target = (coarse, fine) if reverse else (fine, coarse)
            self.prototype_bank.append(PrototypeLayer(
                configs.d_model, source, target,
                exog_dim=configs.load_exog_dim if self.use_exog_prototype else 0))
        self.proto_contrib_ratio = []

    def forward(self, values, exog_list=None):
        ordered = list(reversed(values)) if self.reverse else values
        contexts = list(reversed(exog_list)) if self.reverse and exog_list is not None else exog_list
        current = ordered[0]
        outputs = [current.permute(0, 2, 1)]
        self.proto_contrib_ratio = []
        for index, (bank, target) in enumerate(zip(self.prototype_bank, ordered[1:])):
            batch, hidden, length = current.shape
            query = current.reshape(batch * hidden, length, 1)
            context = None
            if self.use_exog_prototype and contexts is not None:
                context = contexts[index]
                context = context.unsqueeze(1).expand(batch, hidden, length, context.size(-1))
                context = context.reshape(batch * hidden, length, -1)
            answers, _ = bank(query, context)
            residual = answers[-1].reshape(batch, hidden, -1)
            self.proto_contrib_ratio.append((residual.norm() / (target.norm() + 1e-8)).detach())
            current = target + residual
            outputs.append(current.permute(0, 2, 1))
        return list(reversed(outputs)) if self.reverse else outputs


class TemporalMLPMixing(nn.Module):
    """Original TimeMixer scale mappings for the no-prototype ablation."""

    def __init__(self, configs, reverse=False):
        super().__init__()
        self.reverse = reverse
        indices = list(range(configs.down_sampling_layers))
        if reverse:
            indices.reverse()
        mappings = []
        for index in indices:
            fine = configs.seq_len // configs.down_sampling_window ** index
            coarse = fine // configs.down_sampling_window
            source, target = (coarse, fine) if reverse else (fine, coarse)
            mappings.append(nn.Sequential(nn.Linear(source, target), nn.GELU(), nn.Linear(target, target)))
        self.scale_mappings = nn.ModuleList(mappings)
        self.proto_contrib_ratio = []

    def forward(self, values, exog_list=None):
        ordered = list(reversed(values)) if self.reverse else values
        current = ordered[0]
        outputs = [current.permute(0, 2, 1)]
        for mapping, target in zip(self.scale_mappings, ordered[1:]):
            current = target + mapping(current)
            outputs.append(current.permute(0, 2, 1))
        return list(reversed(outputs)) if self.reverse else outputs


class PastDecomposableMixing(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.decompsition = SeriesDecomposition(configs.moving_avg)
        mixing = TemporalMLPMixing if getattr(configs, "variant", "full") == "no_prototype" else MultiScaleMixing
        self.mixing_multi_scale_season = mixing(configs)
        self.mixing_multi_scale_trend = mixing(configs, reverse=True)
        self.out_cross_layer = nn.Sequential(nn.Linear(configs.d_model, configs.d_ff),
                                             nn.GELU(), nn.Linear(configs.d_ff, configs.d_model))
        self.proto_contrib_ratio = []

    def forward(self, values, contexts=None):
        seasonal, trends = [], []
        for value in values:
            season, trend = self.decompsition(value)
            seasonal.append(season.permute(0, 2, 1))
            trends.append(trend.permute(0, 2, 1))
        seasonal = self.mixing_multi_scale_season(seasonal, contexts)
        trends = self.mixing_multi_scale_trend(trends, contexts)
        self.proto_contrib_ratio = (self.mixing_multi_scale_season.proto_contrib_ratio
                                   + self.mixing_multi_scale_trend.proto_contrib_ratio)
        return [original + self.out_cross_layer(season + trend)
                for original, season, trend in zip(values, seasonal, trends)]


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.seq_len, self.pred_len = configs.seq_len, configs.pred_len
        self.load_exog_dim = configs.enc_in - 1 if configs.features == "MS" else 0
        if configs.enc_in != configs.c_out or configs.channel_independence != 1:
            raise ValueError("CMPR-Mixer requires channel_independence=1 and c_out=enc_in")
        if self.load_exog_dim not in (0, 56):
            raise ValueError("use one load channel, or the documented 56 exogenous + load columns")
        if configs.seq_len % configs.down_sampling_window ** configs.down_sampling_layers:
            raise ValueError("seq_len must be divisible by the coarsest down-sampling factor")
        if configs.down_sampling_method not in ("anchor", "avg"):
            raise ValueError("supported pooling: anchor (full model), avg (ablation)")
        self.use_exog_prototype = bool(configs.use_exog_prototype and self.load_exog_dim)
        configs.load_exog_dim = self.load_exog_dim
        configs.use_exog_prototype = int(self.use_exog_prototype)
        self.exog_gate = ExogenousFeatureGate(self.load_exog_dim) if configs.use_exog_gate and self.load_exog_dim else None
        self.pdm_blocks = nn.ModuleList([PastDecomposableMixing(configs) for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(configs.d_model, configs.dropout)
        self.normalize_layers = nn.ModuleList([Normalize(configs.enc_in)
                                               for _ in range(configs.down_sampling_layers + 1)])
        self.anchor_pooling_layers = nn.ModuleList([
            AnchorPooling(configs.enc_in, configs.down_sampling_window)
            for _ in range(configs.down_sampling_layers)] if configs.down_sampling_method == "anchor" else [])
        self.predict_layers = nn.ModuleList([nn.Linear(configs.seq_len // configs.down_sampling_window ** i,
                                                      configs.pred_len)
                                            for i in range(configs.down_sampling_layers + 1)])
        self.projection_layer = nn.Linear(configs.d_model, 1)

    def apply_exogenous_gate(self, values):
        return self.exog_gate(values) if self.exog_gate is not None else values

    def get_exog_gate_weights(self):
        return self.exog_gate.get_weights() if self.exog_gate is not None else None

    def _multiscale_inputs(self, values, marks):
        value_scales, mark_scales = [values], [marks]
        for index in range(self.configs.down_sampling_layers):
            values = values.transpose(1, 2)
            if self.configs.down_sampling_method == "anchor":
                values = self.anchor_pooling_layers[index](values)
            else:
                values = F.avg_pool1d(values, self.configs.down_sampling_window)
            values = values.transpose(1, 2)
            marks = marks[:, ::self.configs.down_sampling_window, :]
            value_scales.append(values)
            mark_scales.append(marks)
        return value_scales, mark_scales

    def forward(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None):
        if x_enc.shape[1:] != (self.seq_len, self.configs.enc_in):
            raise ValueError("input shape must be [batch, seq_len, enc_in]")
        if x_mark_enc is None or x_mark_enc.shape != (*x_enc.shape[:2], 4):
            raise ValueError("hourly time marks must have shape [batch, seq_len, 4]")
        batch, _, channels = x_enc.shape
        values, marks = self._multiscale_inputs(self.apply_exogenous_gate(x_enc), x_mark_enc)
        hidden, contexts = [], [] if self.use_exog_prototype else None
        for index, (value, time) in enumerate(zip(values, marks)):
            length = value.size(1)
            if contexts is not None:
                context = torch.cat([value[:, :, :self.load_exog_dim], time[:, :, :2]], dim=-1)
                contexts.append(context.repeat_interleave(channels, dim=0))
            normalized = self.normalize_layers[index](value, "norm")
            normalized = normalized.permute(0, 2, 1).contiguous().reshape(batch * channels, length, 1)
            if getattr(self.configs, "time_alignment", "legacy") == "legacy":
                time = time.repeat(channels, 1, 1)
            else:
                time = time.repeat_interleave(channels, dim=0)
            hidden.append(self.enc_embedding(normalized, time))
        for block in self.pdm_blocks:
            hidden = block(hidden, contexts)
        predictions = []
        for head, value in zip(self.predict_layers, hidden):
            output = head(value.permute(0, 2, 1)).permute(0, 2, 1)
            if getattr(self.configs, "use_future_temporal_feature", 0):
                if x_mark_dec is None or x_mark_dec.size(1) != self.pred_len:
                    raise ValueError("future calendar marks must have pred_len steps")
                future = (x_mark_dec.repeat(channels, 1, 1)
                          if getattr(self.configs, "time_alignment", "legacy") == "legacy"
                          else x_mark_dec.repeat_interleave(channels, dim=0))
                output = output + self.enc_embedding(None, future)
            output = self.projection_layer(output)
            predictions.append(output.reshape(batch, channels, self.pred_len).permute(0, 2, 1).contiguous())
        return self.normalize_layers[0](torch.stack(predictions, dim=-1).sum(-1), "denorm")
