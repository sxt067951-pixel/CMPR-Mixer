"""Shared source training, target evaluation, and adaptation utilities."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Subset, WeightedRandomSampler

from data_provider.data_loader import Dataset_Custom
from models.CMPRMixer import Model as CMPRMixer
from utils.checkpoints import load_state
from utils.config import ROOT as REPO_ROOT
from utils.losses import FrequencyMSELoss

CITY_IDS = tuple(range(10))
SEQ_LEN = 168
PRED_LEN = 24

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def number_tag(value):
    return str(value).replace("-", "m").replace(".", "p")


def add_common_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--data_root",
        type=Path,
        default=REPO_ROOT / "data" / "transformer",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=REPO_ROOT / "outputs" / "transfer",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--train_epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--d_model", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=32)
    parser.add_argument("--e_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--pooling",
        choices=["anchor", "avg"],
        default="anchor",
    )
    parser.add_argument(
        "--load_loss",
        choices=["mse", "frequency"],
        default="frequency",
    )
    parser.add_argument("--timecf_alpha", type=float, default=0.2)
    parser.add_argument(
        "--use_future_temporal_feature",
        type=int,
        choices=[0, 1],
        default=0,
    )
    parser.add_argument("--random_seed", type=int, default=2021)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="retrain even when a matching checkpoint already exists",
    )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--time_alignment", choices=["legacy", "aligned"], default="legacy")
    parser.add_argument("--dry_run", action="store_true")


def validate_common_args(args):
    if args.batch_size < 1 or args.eval_batch_size < 1:
        raise SystemExit("batch sizes must be positive")
    if args.train_epochs < 1 or args.patience < 1:
        raise SystemExit("training epochs and patience must be positive")
    if args.d_model < 1 or args.d_ff < 1 or args.e_layers < 1:
        raise SystemExit("model dimensions and layer count must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise SystemExit("learning_rate must be positive and finite")
    if not 0.0 <= args.timecf_alpha <= 1.0:
        raise SystemExit("--timecf_alpha must be between 0 and 1")
    if not 0.0 <= args.dropout < 1.0:
        raise SystemExit("--dropout must be in [0, 1)")
    if args.gpu < 0 or args.num_workers < 0:
        raise SystemExit("GPU and worker indices must be non-negative")
    args.data_root = args.data_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    validate_city_files(args.data_root)


def validate_city_ids(city_ids):
    invalid = sorted(set(city_ids) - set(CITY_IDS))
    if invalid:
        raise SystemExit("invalid city ids {}; valid ids are 0..9".format(invalid))


def city_path(data_root: Path, city: int):
    return data_root / "transformer_load_city_{}_exog.csv".format(city)


def validate_city_files(data_root: Path):
    expected_header = None
    for city in CITY_IDS:
        path = city_path(data_root, city)
        if not path.is_file():
            raise SystemExit("city data file not found: {}".format(path))
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            header = next(csv.reader(handle))
        if len(header) != 58 or header[0] != "date" or header[-1] != "load":
            raise SystemExit(
                "{} must contain date + 56 exogenous columns + load".format(path)
            )
        if expected_header is None:
            expected_header = header
        elif header != expected_header:
            raise SystemExit("city files do not share the same column order")


def experiment_tag(args):
    alpha = args.timecf_alpha if args.load_loss == "frequency" else 0.0
    files = []
    for path in sorted(args.data_root.glob("transformer_load_city_*_exog.csv")):
        stat = path.stat()
        files.append((path.name, stat.st_size, stat.st_mtime_ns))
    identity = {**vars(args), "data_root": str(args.data_root), "output_root": None,
                "data_files": files, "implementation": "cmpr-clean-v1"}
    for key in ("force", "dry_run", "targets", "pairs", "direction", "ratios", "strategies"):
        identity.pop(key, None)
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()[:10]
    return (
        "dm{}_df{}_el{}_lr{}_ep{}_bs{}_{}_a{}_{}_ft{}_seed{}".format(
            args.d_model,
            args.d_ff,
            args.e_layers,
            number_tag(args.learning_rate),
            args.train_epochs,
            args.batch_size,
            args.load_loss,
            number_tag(alpha),
            args.pooling,
            args.use_future_temporal_feature,
            args.random_seed,
        ) + "_" + digest
    )


def model_config(args):
    from utils.config import make_model_config
    return make_model_config(args)


def get_device(args):
    if torch.cuda.is_available() and not args.cpu:
        device = torch.device("cuda:{}".format(args.gpu))
        print("Using GPU {}".format(device), flush=True)
    else:
        device = torch.device("cpu")
        print("Using CPU", flush=True)
    return device


def make_city_dataset(args, city: int, flag: str):
    return Dataset_Custom(
        root_path=str(args.data_root) + "/",
        data_path=city_path(args.data_root, city).name,
        flag=flag,
        size=[SEQ_LEN, 0, PRED_LEN],
        features="MS",
        target="load",
        scale=True,
        timeenc=1,
        freq="h",
    )


def make_city_loader(args, city: int, flag: str, ratio=None):
    dataset = make_city_dataset(args, city, flag)
    loader_dataset = dataset
    if ratio is not None:
        if flag != "train":
            raise ValueError("a data ratio can only be applied to the train split")
        keep = max(1, int(math.ceil(len(dataset) * float(ratio))))


        loader_dataset = Subset(dataset, range(len(dataset) - keep, len(dataset)))

    is_train = flag == "train"
    loader = DataLoader(
        loader_dataset,
        batch_size=args.batch_size if is_train else args.eval_batch_size,
        shuffle=is_train,
        num_workers=args.num_workers,
        drop_last=is_train and len(loader_dataset) >= args.batch_size,
        pin_memory=torch.cuda.is_available(),
    )
    return dataset, loader


def make_multi_city_loader(args, cities, flag: str):
    datasets = [make_city_dataset(args, city, flag) for city in cities]
    combined = ConcatDataset(datasets)
    is_train = flag == "train"

    sampler = None
    shuffle = False
    if is_train:
        weights = []
        for dataset in datasets:
            weights.extend([1.0 / len(dataset)] * len(dataset))
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            num_samples=sum(len(dataset) for dataset in datasets),
            replacement=True,
        )

    loader = DataLoader(
        combined,
        batch_size=args.batch_size if is_train else args.eval_batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        drop_last=is_train,
        pin_memory=torch.cuda.is_available(),
    )
    return combined, loader


def build_model(args, device):
    return CMPRMixer(model_config(args)).float().to(device)


def configure_trainable(model, strategy: str):
    if strategy not in {"full", "prototype"}:
        raise ValueError("unknown training strategy: {}".format(strategy))
    for name, parameter in model.named_parameters():
        if strategy == "full":
            parameter.requires_grad = True
        else:
            parameter.requires_grad = (
                "prototype_bank" in name or name.startswith("exog_gate.")
            )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("training strategy selected no parameters")
    return trainable


def training_loss(args):
    if args.load_loss == "frequency":
        return FrequencyMSELoss(args.timecf_alpha)
    return nn.MSELoss()


def forward_target(model, batch, device, use_amp=False):
    batch_x, batch_y, batch_x_mark, batch_y_mark = batch
    batch_x = batch_x.float().to(device, non_blocking=True)
    batch_y = batch_y.float().to(device, non_blocking=True)
    batch_x_mark = batch_x_mark.float().to(device, non_blocking=True)
    batch_y_mark = batch_y_mark.float().to(device, non_blocking=True)
    with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
        output = model(batch_x, batch_x_mark, None, batch_y_mark)
        prediction = output[:, -model.pred_len:, -1:]
        target = batch_y[:, -model.pred_len:, -1:]
    return prediction, target


@torch.no_grad()
def evaluate(model, loader, device, dataset=None, use_amp=False):
    model.eval()
    squared_sum = 0.0
    absolute_sum = 0.0
    value_count = 0
    raw_squared_sum = 0.0
    raw_absolute_sum = 0.0

    load_mean = 0.0
    load_scale = 1.0
    if dataset is not None and hasattr(dataset, "scaler"):
        load_mean = float(dataset.scaler.mean_[-1])
        load_scale = float(dataset.scaler.scale_[-1])

    for batch in loader:
        prediction, target = forward_target(model, batch, device, use_amp)
        error = prediction.float() - target.float()
        squared_sum += torch.sum(error ** 2).item()
        absolute_sum += torch.sum(torch.abs(error)).item()
        value_count += error.numel()

        raw_error = error * load_scale
        raw_squared_sum += torch.sum(raw_error ** 2).item()
        raw_absolute_sum += torch.sum(torch.abs(raw_error)).item()

    if value_count == 0:
        raise RuntimeError("evaluation loader is empty")
    mse = squared_sum / value_count
    mae = absolute_sum / value_count
    raw_mse = raw_squared_sum / value_count
    raw_mae = raw_absolute_sum / value_count
    return {
        "mse": mse,
        "mae": mae,
        "rmse": math.sqrt(mse),
        "raw_mse": raw_mse,
        "raw_mae": raw_mae,
        "raw_rmse": math.sqrt(raw_mse),
        "load_mean": load_mean,
        "load_scale": load_scale,
    }


def train_or_load(
    args,
    train_loader,
    val_loader,
    checkpoint_path: Path,
    device,
    init_checkpoint: Path = None,
    strategy: str = "full",
    learning_rate: float = None,
    train_epochs: int = None,
    patience: int = None,
    force: bool = None,
):
    set_seed(args.random_seed)
    model = build_model(args, device)
    effective_force = args.force if force is None else force
    if checkpoint_path.is_file() and not effective_force:
        print("Loading cached checkpoint: {}".format(checkpoint_path), flush=True)
        load_state(model, checkpoint_path, device)
        return model

    if init_checkpoint is not None:
        if not init_checkpoint.is_file():
            raise FileNotFoundError("source checkpoint not found: {}".format(init_checkpoint))
        load_state(model, init_checkpoint, device)

    trainable = configure_trainable(model, strategy)
    if len(train_loader) == 0:
        raise ValueError("training loader is empty; lower batch_size or provide more training windows")
    lr = args.learning_rate if learning_rate is None else learning_rate
    epochs = args.train_epochs if train_epochs is None else train_epochs
    stop_patience = args.patience if patience is None else patience
    optimizer = torch.optim.Adam(trainable, lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.2,
    ) if epochs * len(train_loader) >= 10 else None
    criterion = training_loss(args)
    amp_enabled = args.use_amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_mse = float("inf")
    bad_epochs = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        sample_count = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            prediction, target = forward_target(model, batch, device, amp_enabled)
            loss = criterion(prediction, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            batch_size = target.size(0)
            loss_sum += loss.item() * batch_size
            sample_count += batch_size

        val_metrics = evaluate(model, val_loader, device, use_amp=amp_enabled)
        train_average = loss_sum / max(sample_count, 1)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_average,
                "val_mse": val_metrics["mse"],
                "val_mae": val_metrics["mae"],
            }
        )
        print(
            "Epoch {:>3}/{} | Train Loss={:.7f} | Val MSE={:.7f} "
            "Val MAE={:.7f} | LR={:.3e}".format(
                epoch,
                epochs,
                train_average,
                val_metrics["mse"],
                val_metrics["mae"],
                optimizer.param_groups[0]["lr"],
            ),
            flush=True,
        )

        if val_metrics["mse"] < best_mse:
            best_mse = val_metrics["mse"]
            bad_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": vars(model_config(args)),
                    "best_val_mse": best_mse,
                    "strategy": strategy,
                },
                str(checkpoint_path),
            )
        else:
            bad_epochs += 1
            if bad_epochs >= stop_patience:
                print("Early stopping", flush=True)
                break

    with (checkpoint_path.parent / "training_history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, ensure_ascii=False, indent=2)
    load_state(model, checkpoint_path, device)
    from utils.artifacts import export_artifacts
    export_artifacts(model, checkpoint_path.parent)
    return model


def source_checkpoint_path(args, city: int):
    return (
        args.output_root
        / "source_models"
        / experiment_tag(args)
        / "city_{}".format(city)
        / "checkpoint.pth"
    )


def train_single_city_source(args, city: int, device):
    """Train once on a source city, or load its matching cached checkpoint."""
    _, train_loader = make_city_loader(args, city, "train")
    _, val_loader = make_city_loader(args, city, "val")
    checkpoint = source_checkpoint_path(args, city)
    model = train_or_load(
        args,
        train_loader,
        val_loader,
        checkpoint,
        device,
        strategy="full",
    )
    return model, checkpoint


def release_model(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

