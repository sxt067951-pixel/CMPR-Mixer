"""Directed transfer between high- and lower-correlation city pairs."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


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
    train_single_city_source,
    validate_city_ids,
    validate_common_args,
    write_csv,
)


DEFAULT_PAIR_SPECS = [
    (6, 7, "high", 0.9712),
    (7, 6, "high", 0.9712),
    (4, 9, "high", 0.9686),
    (9, 4, "high", 0.9686),
    (1, 2, "high", 0.9645),
    (2, 1, "high", 0.9645),
    (0, 3, "low", 0.5864),
    (3, 0, "low", 0.5864),
    (0, 1, "low", 0.6254),
    (1, 0, "low", 0.6254),
    (0, 2, "low", 0.6325),
    (2, 0, "low", 0.6325),
]


def parse_pair(raw):
    normalized = raw.replace("->", ":").replace("→", ":")
    pieces = normalized.split(":")
    if len(pieces) != 2:
        raise argparse.ArgumentTypeError("use SOURCE:TARGET, for example 6:7")
    try:
        source, target = (int(piece.strip()) for piece in pieces)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("city ids must be integers") from exc
    if source == target:
        raise argparse.ArgumentTypeError("source and target cities must differ")
    validate_city_ids([source, target])
    return source, target


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare high- and low-similarity city transfer pairs."
    )
    add_common_arguments(parser)
    parser.add_argument(
        "--pairs",
        type=parse_pair,
        nargs="*",
        default=None,
        help="custom directed pairs such as --pairs 6:7 7:6",
    )
    return parser.parse_args()


def selected_specs(custom_pairs):
    if custom_pairs is None:
        return list(DEFAULT_PAIR_SPECS)
    known = {(source, target): (group, corr) for source, target, group, corr in DEFAULT_PAIR_SPECS}
    specs = []
    for source, target in custom_pairs:
        group, corr = known.get((source, target), ("custom", float("nan")))
        spec = (source, target, group, corr)
        if spec not in specs:
            specs.append(spec)
    return specs


def main():
    args = parse_args()
    validate_common_args(args)
    specs = selected_specs(args.pairs)
    correlations = {}
    for source, target, _, _ in specs:
        for city in (source, target):
            if city not in correlations:
                frame = pd.read_csv(args.data_root / f"transformer_load_city_{city}_exog.csv", usecols=["date", "load"])
                frame = frame.iloc[:int(len(frame) * .7)].set_index("date")["load"]
                correlations[city] = frame
    specs = [(source, target, group, float(correlations[source].corr(correlations[target])))
             for source, target, group, _ in specs]
    if not specs:
        raise SystemExit("select at least one source:target pair")

    needed_cities = sorted(
        {source for source, _, _, _ in specs}
        | {target for _, target, _, _ in specs}
    )
    tag = experiment_tag(args)
    result_root = args.output_root / "similarity" / tag
    print(
        "Similarity-transfer plan: {} directions, source/self models for cities {}".format(
            len(specs), needed_cities
        ),
        flush=True,
    )
    for source, target, group, corr in specs:
        print(
            "  {}: City {} -> City {} (train correlation={})".format(
                group, source, target,
                "N/A" if np.isnan(corr) else "{:.4f}".format(corr),
            )
        )
    if args.dry_run:
        print("Results directory: {}".format(result_root))
        return

    device = get_device(args)
    source_checkpoints = {}
    self_metrics = {}


    for index, city in enumerate(needed_cities, start=1):
        print(
            "\n[Source {}/{}] Training/loading City {}".format(
                index, len(needed_cities), city
            ),
            flush=True,
        )
        model, checkpoint = train_single_city_source(args, city, device)
        test_dataset, test_loader = make_city_loader(args, city, "test")
        self_metrics[city] = evaluate(
            model, test_loader, device, test_dataset, args.use_amp
        )
        source_checkpoints[city] = checkpoint
        release_model(model)

    rows = []
    for index, (source, target, group, corr) in enumerate(specs, start=1):
        print(
            "\n[Transfer {}/{}] {} City {} -> City {}".format(
                index, len(specs), group, source, target
            ),
            flush=True,
        )
        model = build_model(args, device)
        load_state(model, source_checkpoints[source], device)
        target_dataset, target_loader = make_city_loader(args, target, "test")
        metrics = evaluate(
            model, target_loader, device, target_dataset, args.use_amp
        )
        target_self_mse = self_metrics[target]["mse"]
        transfer_ratio = metrics["mse"] / target_self_mse if target_self_mse > 0 else float("nan")
        row = {
            "group": group,
            "source": source,
            "target": target,
            "train_correlation": corr,
            **metrics,
            "target_self_mse": target_self_mse,
            "transfer_ratio": transfer_ratio,
        }
        rows.append(row)
        print(
            "City {} -> City {}: MSE={:.6f} MAE={:.6f} RMSE={:.6f} "
            "R={:.4f}".format(
                source,
                target,
                metrics["mse"],
                metrics["mae"],
                metrics["rmse"],
                transfer_ratio,
            ),
            flush=True,
        )
        release_model(model)

    fields = [
        "group", "source", "target", "train_correlation",
        "mse", "mae", "rmse", "raw_mse", "raw_mae", "raw_rmse",
        "load_mean", "load_scale", "target_self_mse", "transfer_ratio",
    ]
    summary_path = result_root / "similarity_transfer_summary.csv"
    write_csv(summary_path, rows, fields)

    print("\nFinal similarity-transfer summary:")
    for row in rows:
        print(
            "  [{:<6}] City {} -> City {}: MSE={:.6f} MAE={:.6f} "
            "RMSE={:.6f} R={:.4f}".format(
                row["group"], row["source"], row["target"], row["mse"],
                row["mae"], row["rmse"], row["transfer_ratio"]
            )
        )
    for group in ("high", "low"):
        group_rows = [row for row in rows if row["group"] == group]
        if group_rows:
            print(
                "  {}-similarity average: MSE={:.6f} R={:.4f}".format(
                    group,
                    float(np.mean([row["mse"] for row in group_rows])),
                    float(np.mean([row["transfer_ratio"] for row in group_rows])),
                )
            )
    print("Summary saved to: {}".format(summary_path))


if __name__ == "__main__":
    main()
