import os, json, argparse, logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

from ml.ml_config import FEATURES, TARGET_LR, MODELS_DIR, REPORTS_DIR, XGB_PARAMS, FEATURE_LIST_JSON, timestamp_tag
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
    required = FEATURES + [TARGET_LR]
    _validate_dataframe(df, required)

    X = df[FEATURES].copy().values
    y = df[TARGET_LR].astype(int).values

    train_idx, test_idx, _ = time_based_split(df)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    pos = (y_train == 1).sum(); neg = (y_train == 0).sum()
    spw = float(neg)/max(float(pos), 1.0)
    params = dict(XGB_PARAMS); params.setdefault("scale_pos_weight", spw)
    model = XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric="auc", verbose=False, early_stopping_rounds=50)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    f1 = f1_score(y_test, y_pred, average="macro") if len(np.unique(y_test))>1 else float("nan")
    roc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test))>1 else float("nan")
    pr_auc = average_precision_score(y_test, y_prob) if len(np.unique(y_test))>1 else float("nan")
    brier = brier_score_loss(y_test, y_prob) if len(np.unique(y_test))>1 else float("nan")

    cm = confusion_matrix(y_test, y_pred, labels=[0,1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
    disp.plot(); plt.tight_layout(); plt.savefig(os.path.join(REPORTS_DIR, "confusion_matrix_xgb.png")); plt.close()

    tag = timestamp_tag()
    out_ver = os.path.join(MODELS_DIR, f"xgb_pipeline_{tag}.joblib")
    joblib.dump({"model": model, "features": list(FEATURES)}, out_ver)
    joblib.dump({"model": model, "features": list(FEATURES)}, os.path.join(MODELS_DIR, "xgb_pipeline.joblib"))

    with open(os.path.join(MODELS_DIR, FEATURE_LIST_JSON), "w", encoding="utf-8") as f:
        json.dump(list(FEATURES), f)

    metrics = {"f1_macro": f1, "roc_auc": roc, "pr_auc": pr_auc, "brier": brier}
    with open(os.path.join(REPORTS_DIR, "metrics_xgb.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    LOGGER.info(f"Saved XGB -> {out_ver} | metrics={metrics}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/sample_data.csv")
    args = parser.parse_args()
    main(args.data)
