from __future__ import annotations
from typing import Tuple, Optional, List
import pandas as pd
from ml.ml_config import TEST_SIZE

def time_based_split(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    test_size: float = TEST_SIZE,
    val_size: float = 0.0,
) -> Tuple[pd.Index, pd.Index, Optional[pd.Index]]:
    d = df.copy()
    if timestamp_col in d.columns:
        try:
            ts = pd.to_datetime(d[timestamp_col], errors="coerce")
            d = d.loc[ts.notna()].assign(__ts=ts).sort_values("__ts")
        except Exception:
            d = d.reset_index(drop=True)
    else:
        d = d.reset_index(drop=True)
    n = len(d)
    n_test = int(round(n * test_size))
    n_val = int(round(n * val_size))
    n_train = n - n_test - n_val
    if n_train <= 0:
        raise ValueError("Not enough rows for requested split sizes.")
    train_idx = d.index[:n_train]
    val_idx = d.index[n_train:n_train+n_val] if n_val > 0 else None
    test_idx = d.index[n_train+n_val:]
    return train_idx, test_idx, val_idx

def time_window_split(df: pd.DataFrame, timestamp_col: str, train_end: str, val_end: str):
    d = df.copy()
    ts = pd.to_datetime(d[timestamp_col], errors="coerce")
    d = d.loc[ts.notna()].assign(__ts=ts).sort_values("__ts")
    train = d[d["__ts"] <= pd.to_datetime(train_end)]
    val = d[(d["__ts"] > pd.to_datetime(train_end)) & (d["__ts"] <= pd.to_datetime(val_end))]
    test = d[d["__ts"] > pd.to_datetime(val_end)]
    return train.index, test.index, val.index

def spatio_temporal_holdout(df: pd.DataFrame, region_col: str, holdout_regions: List[str], timestamp_col: str = "timestamp", test_size: float = 0.2):
    in_holdout = df[df[region_col].isin(holdout_regions)].index
    non_holdout = df[~df[region_col].isin(holdout_regions)]
    tr_idx, te_idx, _ = time_based_split(non_holdout, timestamp_col=timestamp_col, test_size=test_size, val_size=0.0)
    te_idx = te_idx.union(in_holdout)
    return tr_idx, te_idx, None
