# CMPR-Mixer

[中文说明](README_zh.md)

Code for *Electric Load Forecasting Using Context-Guided Multiscale Pattern Retrieval*.

## Overview

CMPR-Mixer uses 168 hours of historical load and external data to forecast electricity demand for the next 24 hours. Two periods with similar average demand can have very different peaks, and similar load curves can evolve differently on a hot day or a public holiday. To retain local peaks when downsampling, the model learns a correction to the window mean based on the difference between the maximum and the mean. It then uses the recent load, weather, and calendar conditions to retrieve patterns from a learned dictionary. These patterns guide how seasonal fluctuations and longer-term trends are combined across time scales. During training, a frequency-domain loss complements the hourly prediction error to capture the shape of the 24-hour load curve.

We evaluated CMPR-Mixer on hourly data from 369 transformers aggregated by city across 10 cities in Guangxi, China. When models were trained and tested within each city, CMPR-Mixer achieved the lowest average errors among the models compared (MSE **0.1454**, MAE **0.2711**). To test whether the learned patterns transfer to a new city, we trained on nine cities, froze the model, and evaluated it on the tenth. With each city held out in turn, the average MSE and MAE were **0.1332** and **0.2505**, respectively. All four values are based on standardized load and give each city equal weight.

![CMPR-Mixer architecture showing multiscale load decomposition, context-guided pattern retrieval, and seasonal and trend mixing][cmpr-architecture]

*Figure 5 from the paper. The seasonal branch transfers information from fine to coarse scales, and the trend branch transfers it from coarse to fine scales.*

## Installation

From the repository root, create a Python environment and install the pinned dependencies.

```bash
conda create -n cmpr-mixer python=3.9 -y
conda activate cmpr-mixer
python -m pip install -r requirements.txt
```

## Data

Download **Transformer_DB.db** from the [Figshare dataset](https://doi.org/10.6084/m9.figshare.26333452.v1), place it in `data/raw/`, and generate the hourly CSV files.

```bash
python prepare_data.py --db_path data/raw/Transformer_DB.db --output_dir data/transformer
```

The forecasting inputs are `transformer_load_city_0_exog.csv` through `transformer_load_city_9_exog.csv` and `transformer_load_all_exog.csv` for total load. Each row contains a timestamp, 56 ordered external variables, and the load target. If you already have the prepared files, pass their directory with `--data_root`. [Column definitions](data/README.md) · [Dataset paper](https://doi.org/10.1038/s41597-025-04766-7)

## Train and evaluate

Start with one city. This command trains the model, selects the checkpoint with the lowest validation load MSE, and evaluates it on the test split.

```bash
python run.py --cities 0 --skip_total --save_predictions
```

To run the ten within-city experiments used for the city average, use the following command. Omit `--skip_total` to also train a separate model for total load.

```bash
python run.py --all_cities --skip_total
```

The default split is chronological (70% training, 10% validation, 20% testing). To evaluate the saved city 0 checkpoint again, run `python run.py --cities 0 --skip_total --mode test` with the same data and model settings. Use `--cpu` for CPU execution. Paper runs use the default `--time_alignment legacy`. Changing the alignment changes the experiment.

## Ablation and cross-city transfer

Compare the full model with four variants on cities 0–4, the five cities used in the paper's ablation table.

```bash
python run.py --cities 0 1 2 3 4 --skip_total --variants full no_gate avg_pool no_prototype mse_loss --output_root outputs/ablation
```

| Transfer experiment | Command |
| --- | --- |
| Train on nine cities and test on the held-out city, over ten folds | `python transfer/city_loco_transfer.py` |
| Evaluate both directions for city pairs grouped by historical load correlation | `python transfer/city_similarity_transfer.py` |
| Train on cities 0–4 and test on cities 5–9, then reverse | `python transfer/city_group_transfer.py --direction both` |

Transfer experiments require all ten city CSVs. In these evaluations the transferred model is frozen on the target city. Each city's training period supplies its own normalization statistics.

To fine-tune on progressively larger subsets of target-city training windows, run:

```bash
python transfer/city_finetune_transfer.py --ratios 0 5 10 20
```

## Outputs

Forecast runs write a summary CSV to `outputs/forecast/` and save `checkpoint.pth` and `metrics.json` in each run's subdirectory. Use `mse` and `mae` for the standardized load errors reported in the paper. The `raw_*` fields use load values converted back to the original units. With `--save_predictions`, the same directory also contains `predictions.npy` and `targets.npy`. Transfer summaries are under `outputs/transfer/`.

## Acknowledgments

CMPR-Mixer builds on TimeMixer and draws on ProtoTS for prototype retrieval. [Third-party notices](NOTICE)

[cmpr-architecture]: figures/CMPR-Mixer-architecture.png
