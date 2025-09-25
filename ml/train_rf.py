import os, json, argparse, logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ml.ml_config import FEATURES, TARGET_RF, MODELS_DIR, REPORTS_DIR, RF_PARAMS, FEATURE_LIST_JSON, timestamp_tag
from ml.split import time_based_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
LOGGER = logging.getLogger(__name__)

def _validate_dataframe(df, required_cols):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    for c in required_cols:
        if df[c].isna().any():
            LOGGER.warning(f"Column '{c}' contains NaN values; consider imputing or dropping.")

def main(data_path: str = "data/sample_data.csv"):
    Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
    Path(REPORTS_DIR).mkdir(parents=True, exist_ok=True)
    LOGGER.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    required = FEATURES + [TARGET_RF]
    _validate_dataframe(df, required)

    X = df[FEATURES].copy()
    y = df[TARGET_RF].astype(float).values

    train_idx, test_idx, _ = time_based_split(df)
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    pipe = Pipeline([("scaler", StandardScaler()), ("rf", RandomForestRegressor(**RF_PARAMS))])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    tag = timestamp_tag()
    out_ver = os.path.join(MODELS_DIR, f"rf_pipeline_{tag}.joblib")
    joblib.dump(pipe, out_ver)
    joblib.dump(pipe, os.path.join(MODELS_DIR, "rf_pipeline.joblib"))

    with open(os.path.join(MODELS_DIR, FEATURE_LIST_JSON), "w", encoding="utf-8") as f:
        json.dump(FEATURES, f)

    metrics = {"mae": mae, "rmse": rmse, "r2": r2}
    with open(os.path.join(REPORTS_DIR, "metrics_rf.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    LOGGER.info(f"Saved RF model -> {out_ver} | metrics={metrics}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/sample_data.csv")
    args = parser.parse_args()
    main(args.data)
