import os, json, argparse, logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ml.ml_config import FEATURES, TARGET_RF, MODELS_DIR, REPORTS_DIR, RF_GRID, FEATURE_LIST_JSON, timestamp_tag
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

    df = pd.read_csv(data_path)
    required = FEATURES + [TARGET_RF]
    _validate_dataframe(df, required)

    X = df[FEATURES].copy()
    y = df[TARGET_RF].astype(float).values
    train_idx, test_idx, _ = time_based_split(df)
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    pipe = Pipeline([("scaler", StandardScaler()), ("rf", RandomForestRegressor())])
    gs = GridSearchCV(pipe, RF_GRID, cv=3, n_jobs=-1, verbose=1, scoring="neg_root_mean_squared_error", refit=True)
    gs.fit(X_train, y_train)

    best = gs.best_estimator_
    y_pred = best.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    tag = timestamp_tag()
    out_ver = os.path.join(MODELS_DIR, f"rf_pipeline_{tag}.joblib")
    joblib.dump(best, out_ver)
    joblib.dump(best, os.path.join(MODELS_DIR, "rf_pipeline.joblib"))

    with open(os.path.join(MODELS_DIR, FEATURE_LIST_JSON), "w", encoding="utf-8") as f:
        json.dump(FEATURES, f)

    report = {"best_params": gs.best_params_, "cv_best_score_neg_rmse": float(gs.best_score_), "test_metrics": {"mae": mae, "rmse": rmse, "r2": r2}, "version_tag": tag}
    with open(os.path.join(REPORTS_DIR, "gridsearch_rf_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    LOGGER.info(f"GridSearch done | report={report}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/sample_data.csv")
    args = parser.parse_args()
    main(args.data)
