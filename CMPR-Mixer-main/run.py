"""Run CMPR-Mixer main, ablation, and no-exogenous-variable experiments."""

from utils.config import parse_args


def main():
    args = parse_args()
    cities = list(range(10)) if args.all_cities else sorted(set(args.cities))
    targets = ([] if args.skip_total else ["total"]) + [f"city_{c}" for c in cities]
    if not targets:
        raise SystemExit("select total load or at least one city")
    seeds = args.seeds if args.seeds is not None else [args.random_seed]
    if not seeds:
        raise SystemExit("select at least one seed")
    jobs = [(variant, target, seed) for seed in seeds for variant in args.variants for target in targets]
    print(f"CMPR-Mixer | {args.seq_len} -> {args.pred_len} | jobs={len(jobs)}", flush=True)
    if args.dry_run:
        for variant, target, seed in jobs:
            loss = "mse" if variant == "mse_loss" else args.load_loss
            print(f"{variant} | {target} | seed={seed} | loss={loss}")
        return
    from experiments.forecasting import run_jobs
    run_jobs(args, jobs)


if __name__ == "__main__":
    main()
