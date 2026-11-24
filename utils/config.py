"""Arguments shared by main and ablation experiments."""

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("full", "no_gate", "no_prototype", "avg_pool", "mse_loss", "no_exog")


def make_model_config(args):
    variant = getattr(args, "variant", "full")
    no_exog = variant == "no_exog"
    return SimpleNamespace(
        seq_len=getattr(args, "seq_len", 168), pred_len=getattr(args, "pred_len", 24),
        features="S" if no_exog else "MS", enc_in=1 if no_exog else 57,
        c_out=1 if no_exog else 57, channel_independence=1,
        use_exog_gate=int(variant not in ("no_gate", "no_exog")),
        use_exog_prototype=int(variant not in ("no_prototype", "no_exog")),
        d_model=args.d_model, d_ff=args.d_ff, e_layers=args.e_layers,
        dropout=args.dropout, moving_avg=getattr(args, "moving_avg", 25),
        down_sampling_layers=getattr(args, "down_sampling_layers", 3),
        down_sampling_window=2,
        down_sampling_method="avg" if variant == "avg_pool" else args.pooling,
        use_future_temporal_feature=getattr(args, "use_future_temporal_feature", 0),
        time_alignment=getattr(args, "time_alignment", "legacy"), variant=variant,
    )


def parse_args(argv=None):
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--config", type=Path)
    config_args, _ = preliminary.parse_known_args(argv)
    parser = argparse.ArgumentParser(description="CMPR-Mixer hourly load forecasting", parents=[preliminary])
    parser.add_argument("--data_root", type=Path, default=ROOT / "data" / "transformer")
    parser.add_argument("--output_root", type=Path, default=ROOT / "outputs" / "forecast")
    parser.add_argument("--cities", nargs="*", type=int, default=[])
    parser.add_argument("--all_cities", action="store_true")
    parser.add_argument("--skip_total", action="store_true")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=["full"])
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--random_seed", type=int, default=2021)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=.001)
    parser.add_argument("--train_epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--d_model", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=32)
    parser.add_argument("--e_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=.1)
    parser.add_argument("--seq_len", type=int, default=168)
    parser.add_argument("--pred_len", type=int, default=24)
    parser.add_argument("--moving_avg", type=int, default=25)
    parser.add_argument("--down_sampling_layers", type=int, default=3)
    parser.add_argument("--pooling", choices=["anchor", "avg"], default="anchor")
    parser.add_argument("--load_loss", choices=["mse", "frequency"], default="frequency")
    parser.add_argument("--timecf_alpha", type=float, default=.2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--time_alignment", choices=["legacy", "aligned"], default="legacy")
    parser.add_argument("--use_future_temporal_feature", choices=[0, 1], type=int, default=0)
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    if config_args.config:
        settings = json.loads(config_args.config.read_text(encoding="utf-8"))
        unknown = set(settings) - {action.dest for action in parser._actions}
        if unknown:
            parser.error(f"unknown config keys: {sorted(unknown)}")
        parser.set_defaults(**settings)
    args = parser.parse_args(argv)
    args.data_root, args.output_root = Path(args.data_root).resolve(), Path(args.output_root).resolve()
    for name in ("batch_size", "eval_batch_size", "train_epochs", "patience", "d_model", "d_ff", "e_layers", "seq_len", "pred_len"):
        if getattr(args, name) < 1:
            parser.error(f"{name} must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning_rate must be finite and positive")
    if not 0 <= args.timecf_alpha <= 1 or not 0 <= args.dropout < 1:
        parser.error("invalid frequency weight or dropout")
    if args.num_workers < 0 or args.gpu < 0 or args.down_sampling_layers < 1:
        parser.error("invalid worker, GPU or scale count")
    if args.moving_avg < 1 or args.moving_avg % 2 != 1:
        parser.error("moving_avg must be positive and odd")
    if args.seq_len % 2 ** args.down_sampling_layers:
        parser.error("seq_len must be divisible by 2**down_sampling_layers")
    if not args.variants or any(v not in VARIANTS for v in args.variants):
        parser.error("invalid variants")
    if any(c not in range(10) for c in args.cities):
        parser.error("cities must be in 0..9")
    if args.checkpoint and (args.mode != "test" or args.all_cities or len(args.cities) > 1
                            or (args.cities and not args.skip_total) or len(args.variants) != 1
                            or (args.seeds and len(args.seeds) != 1)):
        parser.error("--checkpoint requires test mode and exactly one target, variant and seed")
    return args
