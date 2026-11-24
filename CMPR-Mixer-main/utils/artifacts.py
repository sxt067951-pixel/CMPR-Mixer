"""Save interpretable model artifacts without bundling training data."""

import json
from pathlib import Path

import torch


def export_artifacts(model, output_dir, columns=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    banks = {}
    for name, module in model.named_modules():
        if not hasattr(module, "prototype_embedding"):
            continue
        banks[name] = {"input_length": module.input_length,
                       "output_length": module.output_length,
                       "keys": module.prototype_embedding[0].detach().cpu(),
                       "values": module.prototype_value[0].detach().cpu(),
                       "state": {k: v.detach().cpu() for k, v in module.state_dict().items()}}
    torch.save(banks, output_dir / "prototypes.pth")
    gate = model.get_exog_gate_weights()
    if gate is not None:
        if columns is not None:
            gate["columns"] = columns[:model.load_exog_dim]
        (output_dir / "exogenous_gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")


def save_scaler(dataset, path):
    payload = {"columns": dataset.columns, "train_rows": dataset.train_rows,
               "mean": dataset.scaler.mean_.tolist(), "scale": dataset.scaler.scale_.tolist(),
               "variance": dataset.scaler.var_.tolist()}
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
