import os, json, argparse, logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from ml.ml_config import FEATURES, TARGET_LR, MODELS_DIR, REPORTS_DIR, LR_PARAMS, FEATURE_LIST_JSON, timestamp_tag
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
    required = FEATURES + [TARGET_LR]
    _validate_dataframe(df, required)

    X = df[FEATURES].copy()
    y = df[TARGET_LR].astype(int).values

    train_idx, test_idx, _ = time_based_split(df)
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(**LR_PARAMS))])
    pipe.fit(X_train, y_train)

    y_prob = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    f1 = f1_score(y_test, y_pred, average="macro") if len(np.unique(y_test))>1 else float("nan")
    roc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test))>1 else float("nan")
    pr_auc = average_precision_score(y_test, y_prob) if len(np.unique(y_test))>1 else float("nan")
    brier = brier_score_loss(y_test, y_prob) if len(np.unique(y_test))>1 else float("nan")

    cm = confusion_matrix(y_test, y_pred, labels=[0,1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
    disp.plot(); plt.tight_layout(); plt.savefig(os.path.join(REPORTS_DIR, "confusion_matrix_lr.png")); plt.close()

    tag = timestamp_tag()
    out_ver = os.path.join(MODELS_DIR, f"lr_pipeline_{tag}.joblib")
    joblib.dump(pipe, out_ver)
    joblib.dump(pipe, os.path.join(MODELS_DIR, "lr_pipeline.joblib"))

    with open(os.path.join(MODELS_DIR, FEATURE_LIST_JSON), "w", encoding="utf-8") as f:
        json.dump(FEATURES, f)

    metrics = {"f1_macro": f1, "roc_auc": roc, "pr_auc": pr_auc, "brier": brier}
    with open(os.path.join(REPORTS_DIR, "metrics_lr.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    LOGGER.info(f"Saved LR model -> {out_ver} | metrics={metrics}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/sample_data.csv")
    args = parser.parse_args()
    main(args.data)
