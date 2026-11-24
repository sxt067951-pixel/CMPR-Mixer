"""Leave-one-city-out transfer: nine source cities to one held-out city."""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.engine import (
    CITY_IDS, add_common_arguments, validate_common_args, validate_city_ids,
    experiment_tag, get_device, make_multi_city_loader, train_or_load,
    make_city_loader, evaluate, release_model, write_csv
)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Leave one city out: train on nine cities and test on the tenth."
    )
    add_common_arguments(parser)
    parser.add_argument(
        "--targets",
        type=int,
        nargs="*",
        default=list(CITY_IDS),
        help="held-out target cities; default: all ten",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    validate_common_args(args)
    validate_city_ids(args.targets)
    targets = sorted(set(args.targets))
    if not targets:
        raise SystemExit("select at least one held-out target city")

    tag = experiment_tag(args)
    experiment_root = args.output_root / "loco" / tag
    print(
        "LOCO plan: {} held-out targets, {} source cities per target, 168 -> 24".format(
            len(targets), len(CITY_IDS) - 1
        ),
        flush=True,
    )
    if args.dry_run:
        for target in targets:
            sources = [city for city in CITY_IDS if city != target]
            print("sources={} -> target={}".format(sources, target))
        print("Results directory: {}".format(experiment_root))
        return

    device = get_device(args)
    rows = []
    for index, target in enumerate(targets, start=1):
        sources = [city for city in CITY_IDS if city != target]
        print(
            "\n[{}/{}] LOCO sources={} -> target={}".format(
                index, len(targets), sources, target
            ),
            flush=True,
        )
        _, train_loader = make_multi_city_loader(args, sources, "train")
        _, val_loader = make_multi_city_loader(args, sources, "val")
        checkpoint = experiment_root / "target_{}".format(target) / "checkpoint.pth"
        model = train_or_load(
            args, train_loader, val_loader, checkpoint, device, strategy="full"
        )
        target_dataset, target_loader = make_city_loader(args, target, "test")
        metrics = evaluate(
            model, target_loader, device, target_dataset, args.use_amp
        )
        row = {
            "target": target,
            "sources": " ".join(map(str, sources)),
            **metrics,
        }
        rows.append(row)
        print(
            "LOCO target City {}: MSE={:.6f} MAE={:.6f} RMSE={:.6f}".format(
                target, metrics["mse"], metrics["mae"], metrics["rmse"]
            ),
            flush=True,
        )
        release_model(model)

    fields = [
        "target", "sources", "mse", "mae", "rmse",
        "raw_mse", "raw_mae", "raw_rmse", "load_mean", "load_scale",
    ]
    summary_path = experiment_root / "loco_summary.csv"
    write_csv(summary_path, rows, fields)
    print("\nFinal LOCO summary:")
    for row in rows:
        print(
            "  sources=all-except-{0} -> City {0}: MSE={1:.6f} "
            "MAE={2:.6f} RMSE={3:.6f}".format(
                row["target"], row["mse"], row["mae"], row["rmse"]
            )
        )
    print(
        "  Macro average: MSE={:.6f} MAE={:.6f} RMSE={:.6f}".format(
            float(np.mean([row["mse"] for row in rows])),
            float(np.mean([row["mae"] for row in rows])),
            float(np.mean([row["rmse"] for row in rows])),
        )
    )
    print("Summary saved to: {}".format(summary_path))


if __name__ == "__main__":
    main()
