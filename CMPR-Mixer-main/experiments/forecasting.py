"""Run total/city experiments and write a persistent final summary."""

import copy
import gc
import hashlib
import json
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_provider.data_loader import Dataset_Custom
from experiments.engine import evaluate, forward_target, get_device, set_seed, train_or_load, write_csv
from models.CMPRMixer import Model
from utils.artifacts import export_artifacts, save_scaler
from utils.checkpoints import load_state
from utils.config import make_model_config


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def make_loader(args, filename, split):
    config = make_model_config(args)
    dataset = Dataset_Custom(args.data_root, data_path=filename, flag=split,
                             features=config.features, size=[args.seq_len, 0, args.pred_len])
    loader = DataLoader(dataset, batch_size=args.batch_size if split == "train" else args.eval_batch_size,
                        shuffle=split == "train", drop_last=split == "train",
                        num_workers=args.num_workers, pin_memory=not args.cpu and torch.cuda.is_available())
    return dataset, loader


def job_directory(args, target, filename):
    stat = (args.data_root / filename).stat()
    identity = {**vars(args), "data_file": str(args.data_root / filename),
                "data_bytes": stat.st_size, "data_mtime_ns": stat.st_mtime_ns,
                "implementation": "cmpr-clean-v1"}
    for key in ("mode", "checkpoint", "config", "force", "dry_run", "output_root", "cities", "all_cities",
                "skip_total", "variants", "seeds", "save_predictions"):
        identity.pop(key, None)
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()[:12]
    return args.output_root / f"{args.variant}_{target}_s{args.random_seed}_{digest}"


def run_single(args, target):
    filename = "transformer_load_all_exog.csv" if target == "total" else f"transformer_load_{target}_exog.csv"
    if not (args.data_root / filename).is_file():
        raise FileNotFoundError(f"Missing {args.data_root / filename}. See data/README.md.")
    directory = job_directory(args, target, filename)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint) if args.checkpoint else directory / "checkpoint.pth"
    model = None
    with (directory / "run.log").open("a", encoding="utf-8") as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            try:
                config = {"target": target, "arguments": vars(args),
                          "model_config": vars(make_model_config(args)),
                          "checkpoint": str(checkpoint), "validation_metric": "load-only MSE"}
                (directory / "config.json").write_text(json.dumps(config, default=str, indent=2), encoding="utf-8")
                device = get_device(args)
                set_seed(args.random_seed)
                test_dataset, test_loader = make_loader(args, filename, "test")
                if args.mode == "train":
                    _, train_loader = make_loader(args, filename, "train")
                    _, val_loader = make_loader(args, filename, "val")
                    model = train_or_load(args, train_loader, val_loader, checkpoint, device)
                else:
                    if not checkpoint.is_file():
                        raise FileNotFoundError(f"No checkpoint: {checkpoint}")
                    model = Model(make_model_config(args)).float().to(device)
                    load_state(model, checkpoint, device)
                metrics = evaluate(model, test_loader, device, test_dataset, args.use_amp)
                metrics["parameters"] = sum(p.numel() for p in model.parameters())
                (directory / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
                export_artifacts(model, directory, test_dataset.columns)
                save_scaler(test_dataset, directory / "scaler.json")
                if args.save_predictions:
                    predictions, truths = [], []
                    with torch.no_grad():
                        for batch in test_loader:
                            prediction, truth = forward_target(model, batch, device, args.use_amp)
                            predictions.append(prediction.cpu().float().numpy())
                            truths.append(truth.cpu().float().numpy())
                    np.save(directory / "predictions.npy", np.concatenate(predictions))
                    np.save(directory / "targets.npy", np.concatenate(truths))
                print(f"Final: {args.variant} {target}: MSE={metrics['mse']:.6f} "
                      f"MAE={metrics['mae']:.6f} RMSE={metrics['rmse']:.6f}", flush=True)
                return metrics, directory
            except Exception:
                traceback.print_exc()
                raise
            finally:
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


def run_jobs(args, jobs):
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary = args.output_root / f"summary_{datetime.now():%Y%m%d_%H%M%S_%f}.csv"
    rows = []
    fields = ["variant", "target", "seed", "status", "mse", "mae", "rmse", "raw_mse", "raw_mae",
              "raw_rmse", "load_mean", "load_scale", "parameters", "directory", "error"]
    for index, (variant, target, seed) in enumerate(jobs, 1):
        current = copy.deepcopy(args)
        current.variant, current.random_seed = variant, seed
        if variant == "mse_loss":
            current.load_loss = "mse"
        print(f"\n[{index}/{len(jobs)}] {variant} | {target} | seed={seed}", flush=True)
        row = {"variant": variant, "target": target, "seed": seed}
        try:
            metrics, directory = run_single(current, target)
            row.update(metrics, status="ok", directory=str(directory), error="")
        except Exception as error:
            row.update(status="failed", error=str(error))
            print(f"FAILED: {error}", file=sys.stderr, flush=True)
        rows.append(row)
        write_csv(summary, rows, fields)
    print("\nFinal experiment summary:")
    for row in rows:
        suffix = (f"MSE={row['mse']:.6f} MAE={row['mae']:.6f} RMSE={row['rmse']:.6f}"
                  if row["status"] == "ok" else f"FAILED: {row['error']}")
        print(f"  {row['variant']} | {row['target']} | seed={row['seed']} | {suffix}")
    print(f"Summary saved to: {summary}")
    if any(row["status"] != "ok" for row in rows):
        raise SystemExit(1)
