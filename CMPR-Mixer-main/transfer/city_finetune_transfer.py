"""Target adaptation using 0%, 5%, 10%, and 20% training windows."""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path


TRANSFER_ROOT = Path(__file__).resolve().parent
REPO_ROOT = TRANSFER_ROOT.parent
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from experiments.engine import (
    add_common_arguments,
    build_model,
    evaluate,
    experiment_tag,
    get_device,
    load_state,
    make_city_loader,
    release_model,
    train_or_load,
    train_single_city_source,
    validate_city_ids,
    validate_common_args,
    write_csv,
)


DEFAULT_PAIRS = [(6, 7), (7, 6), (0, 3), (3, 0)]


def parse_pair(raw):
    normalized = raw.replace("->", ":").replace("→", ":")
    pieces = normalized.split(":")
    if len(pieces) != 2:
        raise argparse.ArgumentTypeError("use SOURCE:TARGET, for example 6:7")
    try:
        source, target = (int(piece.strip()) for piece in pieces)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("city ids must be integers") from exc
    validate_city_ids([source, target])
    if source == target:
        raise argparse.ArgumentTypeError("source and target cities must differ")
    return source, target


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare zero-shot and gradual target-city fine-tuning."
    )
    add_common_arguments(parser)
    parser.add_argument(
        "--pairs",
        type=parse_pair,
        nargs="*",
        default=list(DEFAULT_PAIRS),
        help="representative directions; default: 6:7 7:6 0:3 3:0",
    )
    parser.add_argument(
        "--ratios",
        type=int,
        nargs="*",
        default=[0, 5, 10, 20],
        help="percentages of target training windows; default stops at 20%%",
    )
    parser.add_argument(
        "--strategies",
        nargs="*",
        choices=["prototype", "full", "scratch"],
        default=["prototype", "full", "scratch"],
    )
    parser.add_argument("--finetune_learning_rate", type=float, default=0.0001)
    parser.add_argument("--scratch_learning_rate", type=float, default=0.001)
    parser.add_argument("--finetune_epochs", type=int, default=10)
    parser.add_argument("--finetune_patience", type=int, default=3)
    return parser.parse_args()


def trainable_parameter_counts(model, strategy):
    total = sum(parameter.numel() for parameter in model.parameters())
    if strategy == "prototype":
        trainable = sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if "prototype_bank" in name or name.startswith("exog_gate.")
        )
    elif strategy in {"full", "scratch"}:
        trainable = total
    else:
        trainable = 0
    return trainable, total


def main():
    args = parse_args()
    validate_common_args(args)
    if not args.pairs:
        raise SystemExit("select at least one source:target pair")
    if not args.ratios:
        raise SystemExit("select at least one fine-tuning ratio")
    if any(ratio < 0 or ratio > 20 for ratio in args.ratios):
        raise SystemExit("fine-tuning ratios must be between 0 and 20")
    if any(not math.isfinite(value) or value <= 0 for value in
           (args.finetune_learning_rate, args.scratch_learning_rate)):
        raise SystemExit("fine-tuning learning rates must be positive")
    if args.finetune_epochs < 1 or args.finetune_patience < 1:
        raise SystemExit("fine-tuning epochs and patience must be positive")

    pairs = []
    for pair in args.pairs:
        if pair not in pairs:
            pairs.append(pair)
    ratios = sorted(set(args.ratios))
    strategies = list(dict.fromkeys(args.strategies))
    tag = experiment_tag(args)
    fine_tag = "ftlr{}_scratchlr{}_ftep{}".format(
        str(args.finetune_learning_rate).replace(".", "p"),
        str(args.scratch_learning_rate).replace(".", "p"),
        args.finetune_epochs,
    )
    result_root = args.output_root / "finetune" / tag / fine_tag

    jobs_per_pair = int(0 in ratios) + len([r for r in ratios if r > 0]) * len(strategies)
    print(
        "Fine-tuning plan: pairs={} ratios={} strategies={} total evaluations={}".format(
            pairs, ratios, strategies, len(pairs) * jobs_per_pair
        ),
        flush=True,
    )
    if args.dry_run:
        for source, target in pairs:
            print("City {} -> City {}: zero-shot + ratios {}".format(source, target, ratios))
        print("Results directory: {}".format(result_root))
        return

    device = get_device(args)
    source_checkpoints = {}
    for index, source in enumerate(sorted({source for source, _ in pairs}), start=1):
        print("\n[Source {}] Training/loading City {}".format(index, source), flush=True)
        model, checkpoint = train_single_city_source(args, source, device)
        source_checkpoints[source] = checkpoint
        release_model(model)

    rows = []
    for pair_index, (source, target) in enumerate(pairs, start=1):
        print(
            "\n[Pair {}/{}] City {} -> City {}".format(
                pair_index, len(pairs), source, target
            ),
            flush=True,
        )
        target_test_dataset, target_test_loader = make_city_loader(args, target, "test")

        if 0 in ratios:
            model = build_model(args, device)
            load_state(model, source_checkpoints[source], device)
            metrics = evaluate(
                model, target_test_loader, device, target_test_dataset, args.use_amp
            )
            trainable, total = trainable_parameter_counts(model, "zero_shot")
            rows.append(
                {
                    "source": source,
                    "target": target,
                    "ratio_percent": 0,
                    "strategy": "zero_shot",
                    "target_train_windows": 0,
                    "trainable_parameters": trainable,
                    "total_parameters": total,
                    **metrics,
                }
            )
            print(
                "  0% zero_shot: MSE={:.6f} MAE={:.6f} RMSE={:.6f}".format(
                    metrics["mse"], metrics["mae"], metrics["rmse"]
                ),
                flush=True,
            )
            release_model(model)

        for ratio in (value for value in ratios if value > 0):
            base_train_dataset, target_train_loader = make_city_loader(
                args, target, "train", ratio=ratio / 100.0
            )
            target_train_windows = len(target_train_loader.dataset)
            _, target_val_loader = make_city_loader(args, target, "val")

            for strategy in strategies:
                init_checkpoint = (
                    None if strategy == "scratch" else source_checkpoints[source]
                )
                train_strategy = "full" if strategy == "scratch" else strategy
                lr = (
                    args.scratch_learning_rate
                    if strategy == "scratch"
                    else args.finetune_learning_rate
                )
                checkpoint = (
                    result_root
                    / "city_{}_to_{}".format(source, target)
                    / "ratio_{}".format(ratio)
                    / strategy
                    / "checkpoint.pth"
                )
                print(
                    "  {}% {} | train windows={} | lr={}".format(
                        ratio, strategy, target_train_windows, lr
                    ),
                    flush=True,
                )
                model = train_or_load(
                    args,
                    target_train_loader,
                    target_val_loader,
                    checkpoint,
                    device,
                    init_checkpoint=init_checkpoint,
                    strategy=train_strategy,
                    learning_rate=lr,
                    train_epochs=args.finetune_epochs,
                    patience=args.finetune_patience,
                )
                metrics = evaluate(
                    model,
                    target_test_loader,
                    device,
                    target_test_dataset,
                    args.use_amp,
                )
                trainable, total = trainable_parameter_counts(model, strategy)
                rows.append(
                    {
                        "source": source,
                        "target": target,
                        "ratio_percent": ratio,
                        "strategy": strategy,
                        "target_train_windows": target_train_windows,
                        "trainable_parameters": trainable,
                        "total_parameters": total,
                        **metrics,
                    }
                )
                print(
                    "  Final {}% {}: MSE={:.6f} MAE={:.6f} RMSE={:.6f}".format(
                        ratio, strategy, metrics["mse"], metrics["mae"], metrics["rmse"]
                    ),
                    flush=True,
                )
                release_model(model)

    fields = [
        "source", "target", "ratio_percent", "strategy",
        "target_train_windows", "trainable_parameters", "total_parameters",
        "mse", "mae", "rmse", "raw_mse", "raw_mae", "raw_rmse",
        "load_mean", "load_scale",
    ]
    summary_path = result_root / "finetune_summary.csv"
    write_csv(summary_path, rows, fields)

    print("\nFinal fine-tuning summary:")
    for row in rows:
        print(
            "  City {} -> City {} | {:>2}% {:<10}: MSE={:.6f} "
            "MAE={:.6f} RMSE={:.6f}".format(
                row["source"], row["target"], row["ratio_percent"],
                row["strategy"], row["mse"], row["mae"], row["rmse"]
            )
        )
    print("Summary saved to: {}".format(summary_path))


if __name__ == "__main__":
    main()
