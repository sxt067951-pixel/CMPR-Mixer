"""Bidirectional transfer between City 0..4 and City 5..9."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


TRANSFER_ROOT = Path(__file__).resolve().parent
REPO_ROOT = TRANSFER_ROOT.parent

if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from experiments.engine import (
    PRED_LEN,
    SEQ_LEN,
    add_common_arguments,
    city_path,
    evaluate,
    experiment_tag,
    get_device,
    make_city_loader,
    make_multi_city_loader,
    model_config,
    release_model,
    train_or_load,
    validate_common_args,
    write_csv,
)


FIRST_FIVE = (0, 1, 2, 3, 4)
LAST_FIVE = (5, 6, 7, 8, 9)
DIRECTIONS = {
    "first_to_last": (FIRST_FIVE, LAST_FIVE),
    "last_to_first": (LAST_FIVE, FIRST_FIVE),
}
METRIC_FIELDS = (
    "mse", "mae", "rmse", "raw_mse", "raw_mae", "raw_rmse",
    "load_mean", "load_scale",
)
SUMMARY_FIELDS = (
    "direction", "sources", "target", "seq_len", "pred_len",
    *METRIC_FIELDS, "checkpoint",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train City 0..4 -> test City 5..9, then reverse the groups."
    )
    add_common_arguments(parser)
    parser.add_argument(
        "--direction",
        choices=["both", *DIRECTIONS],
        default="both",
        help="default: train both source groups, evaluating ten held-out cities",
    )
    return parser.parse_args(argv)


def cache_config(args):
    """Include parameters missing from the old tag to avoid stale checkpoints."""
    files = []
    for city in FIRST_FIVE + LAST_FIVE:
        path = city_path(args.data_root, city)
        stat = path.stat()
        files.append({"city": city, "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return {
        "protocol_version": 2,
        "data_root": str(args.data_root),
        "data_files": files,
        "model_config": vars(model_config(args)),
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "train_epochs": args.train_epochs,
        "patience": args.patience,
        "load_loss": args.load_loss,
        "timecf_alpha": args.timecf_alpha,
        "random_seed": args.random_seed,
        "use_amp": args.use_amp,
        "eval_batch_size": args.eval_batch_size,
        "source_groups": {name: list(groups[0]) for name, groups in DIRECTIONS.items()},
    }


def source_label(cities):
    return "City [{}]".format(", ".join(map(str, cities)))


def summary_lines(rows):
    lines = ["Final test summary:"]
    for row in rows:
        lines.append(
            "  sources={sources} -> City {target} | {seq_len} -> {pred_len}: "
            "Final: MSE={mse:.6f} MAE={mae:.6f} RMSE={rmse:.6f}".format(**row)
        )
    for direction, (sources, _) in DIRECTIONS.items():
        group = [row for row in rows if row["direction"] == direction]
        if group:
            means = {
                name: sum(row[name] for row in group) / len(group)
                for name in ("mse", "mae", "rmse")
            }
            lines.append(
                "  {} | sources={} | Macro average ({} targets): "
                "MSE={:.6f} MAE={:.6f} RMSE={:.6f}".format(
                    direction, source_label(sources), len(group),
                    means["mse"], means["mae"], means["rmse"],
                )
            )
    if len(rows) > 5:
        means = [sum(row[name] for row in rows) / len(rows) for name in ("mse", "mae", "rmse")]
        lines.append(
            "  Overall macro average ({} targets): MSE={:.6f} MAE={:.6f} "
            "RMSE={:.6f}".format(len(rows), *means)
        )
    return lines


def save_summary(experiment_root, rows, direction="both"):


    stem = "five_city_summary"
    if direction != "both":
        stem += "_" + direction
    csv_path = experiment_root / (stem + ".csv")
    txt_path = experiment_root / (stem + ".txt")
    write_csv(csv_path, rows, SUMMARY_FIELDS)
    txt_path.write_text("\n".join(summary_lines(rows)) + "\n", encoding="utf-8")
    return csv_path, txt_path


def main(argv=None):
    args = parse_args(argv)
    if args.learning_rate <= 0:
        raise SystemExit("--learning_rate must be positive")
    if args.num_workers < 0:
        raise SystemExit("--num_workers must be non-negative")
    validate_common_args(args)
    names = list(DIRECTIONS) if args.direction == "both" else [args.direction]
    config = cache_config(args)
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()[:10]
    experiment_root = args.output_root / "five_city_transfer" / (experiment_tag(args) + "_" + digest)
    total_targets = sum(len(DIRECTIONS[name][1]) for name in names)

    print(
        "Five-city transfer plan: {} source-group trainings, {} target tests | "
        "{} -> {} | CMPR-Mixer | no fine-tuning".format(
            len(names), total_targets, SEQ_LEN, PRED_LEN,
        ), flush=True,
    )
    print(
        "Parameters: batch_size={} eval_batch_size={} learning_rate={} "
        "train_epochs={} patience={} d_model={} d_ff={} e_layers={} "
        "pooling={} load_loss={} alpha={}".format(
            args.batch_size, args.eval_batch_size, args.learning_rate,
            args.train_epochs, args.patience, args.d_model, args.d_ff,
            args.e_layers, args.pooling, args.load_loss, args.timecf_alpha,
        ), flush=True,
    )
    print("Normalization: each city's historical train-split statistics; load-only metrics.", flush=True)
    for name in names:
        sources, targets = DIRECTIONS[name]
        print("  {}: sources={} -> targets={}".format(name, list(sources), list(targets)), flush=True)
    print("Results directory: {}".format(experiment_root), flush=True)
    if args.dry_run:
        return

    experiment_root.mkdir(parents=True, exist_ok=True)
    run_config = {
        **config,
        "selected_directions": names,
        "checkpoint_selection": "source cities' validation splits only",
        "target_adaptation": "none; complete source checkpoint including prototypes is frozen",
        "normalization": "each city's own first 70%; no validation/test fitting",
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (experiment_root / "experiment_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    device = get_device(args)
    rows = []
    completed = 0
    for index, name in enumerate(names, start=1):
        sources, targets = DIRECTIONS[name]
        checkpoint = experiment_root / ("sources_" + "_".join(map(str, sources))) / "checkpoint.pth"
        print(
            "\n[Train {}/{}] sources={} | validation uses these same source cities only".format(
                index, len(names), list(sources),
            ), flush=True,
        )
        train_dataset, train_loader = make_multi_city_loader(args, sources, "train")
        val_dataset, val_loader = make_multi_city_loader(args, sources, "val")
        model = train_or_load(
            args, train_loader, val_loader, checkpoint, device, strategy="full",
        )
        del train_dataset, train_loader, val_dataset, val_loader


        model.requires_grad_(False)
        try:
            for target in targets:
                target_dataset, target_loader = make_city_loader(args, target, "test")
                metrics = evaluate(model, target_loader, device, target_dataset, args.use_amp)
                rows.append({
                    "direction": name,
                    "sources": source_label(sources),
                    "target": target,
                    "seq_len": SEQ_LEN,
                    "pred_len": PRED_LEN,
                    **metrics,
                    "checkpoint": str(checkpoint),
                })
                completed += 1
                print(
                    "[Test {}/{}] sources={} -> City {}: MSE={:.6f} MAE={:.6f} RMSE={:.6f}".format(
                        completed, total_targets, source_label(sources), target,
                        metrics["mse"], metrics["mae"], metrics["rmse"],
                    ), flush=True,
                )

                save_summary(experiment_root, rows, args.direction)
                del target_dataset, target_loader
        finally:
            del model
            release_model(None)

    csv_path, txt_path = save_summary(experiment_root, rows, args.direction)
    print("\n" + "\n".join(summary_lines(rows)), flush=True)
    print("CSV saved to: {}".format(csv_path), flush=True)
    print("Text summary saved to: {}".format(txt_path), flush=True)


if __name__ == "__main__":
    main()
