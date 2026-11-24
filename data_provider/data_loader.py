"""Chronological 70/10/20 splits with training-only standardization."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


def hourly_time_features(dates):
    index = pd.DatetimeIndex(dates)
    return np.stack([index.hour / 23.0 - .5, index.dayofweek / 6.0 - .5,
                     (index.day - 1) / 30.0 - .5, (index.dayofyear - 1) / 365.0 - .5], axis=1)


class Dataset_Custom(Dataset):
    """Load-only targets; validation/test may borrow preceding input history."""

    def __init__(self, root_path, flag="train", size=None, features="MS",
                 data_path="transformer_load_all_exog.csv", target="load", scale=True,
                 timeenc=1, freq="h"):
        if flag not in ("train", "val", "test"):
            raise ValueError("flag must be train, val or test")
        if features not in ("MS", "S") or timeenc != 1 or freq.lower() != "h":
            raise ValueError("supported inputs: features MS/S, timeenc=1, hourly frequency")
        self.seq_len, self.label_len, self.pred_len = size or [168, 0, 24]
        if self.seq_len < 1 or self.pred_len < 1 or not 0 <= self.label_len <= self.seq_len:
            raise ValueError("invalid history, label or forecast lengths")
        self.features, self.target, self.scale = features, target, scale
        self.root_path, self.data_path = str(root_path), str(data_path)
        path = Path(root_path) / data_path
        frame = pd.read_csv(path)
        if "date" not in frame or target not in frame:
            raise ValueError(f"{path} must contain date and {target}")
        external = [name for name in frame if name not in ("date", target)]
        self.columns = external + [target] if features == "MS" else [target]
        if features == "MS" and len(external) != 56:
            raise ValueError("MS input requires exactly 56 exogenous columns; see data/README.md")
        dates = pd.DatetimeIndex(pd.to_datetime(frame["date"], errors="raise"))
        if not dates.is_monotonic_increasing or dates.has_duplicates:
            raise ValueError("dates must be strictly increasing and unique")
        if not np.all(np.diff(dates.asi8) == pd.Timedelta(hours=1).value):
            raise ValueError("data must be recorded once per hour without gaps")
        values = frame[self.columns].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("input contains missing or infinite values; run preprocessing first")
        n = len(frame)
        n_train, n_test = int(n * .7), int(n * .2)
        ends = [n_train, n - n_test, n]
        starts = [0, n_train - self.seq_len, n - n_test - self.seq_len]
        if starts[1] < 0 or min(n_train - self.seq_len, n - n_test - n_train, n_test) < self.pred_len:
            raise ValueError("not enough rows for the configured splits and forecast window")
        split = {"train": 0, "val": 1, "test": 2}[flag]
        self.scaler = StandardScaler().fit(values[:n_train])
        if scale:
            values = self.scaler.transform(values)
        start, end = starts[split], ends[split]
        self.data_x = values[start:end]
        self.data_y = self.data_x
        self.data_stamp = hourly_time_features(dates[start:end])
        self.train_rows = n_train
        self.split_start, self.split_end = start, end
        self.dates = dates[start:end]

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        split = index + self.seq_len
        end = split + self.pred_len
        return (self.data_x[index:split], self.data_y[split - self.label_len:end],
                self.data_stamp[index:split], self.data_stamp[split - self.label_len:end])

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
