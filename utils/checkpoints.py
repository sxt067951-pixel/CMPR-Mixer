"""Explicit compatibility with fixed-capacity legacy load checkpoints."""

import re

import torch


def load_state(model, checkpoint_path, device="cpu"):
    payload = torch.load(str(checkpoint_path), map_location=device, weights_only=True)
    state = payload.get("model_state", payload) if isinstance(payload, dict) else payload
    cleaned = {}
    removed = []
    for name, value in state.items():
        name = name.removeprefix("module.")
        if name == "enc_embedding.position_embedding.pe" or re.fullmatch(r"pdm_blocks\.\d+\.layer_norm\.(weight|bias)", name):
            removed.append(name)
            continue
        cleaned[name] = value
    model.load_state_dict(cleaned, strict=True)
    return removed
