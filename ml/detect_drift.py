import argparse, json
import pandas as pd
from scipy.stats import ks_2samp
from ml.ml_config import FEATURES, REPORTS_DIR

def ks_test_drift(train_df: pd.DataFrame, prod_df: pd.DataFrame, features, alpha=0.05):
    rep = {}
    for f in features:
        a = train_df[f].dropna().values
        b = prod_df[f].dropna().values
        if len(a) == 0 or len(b) == 0:
            rep[f] = {"status": "insufficient_data", "p_value": None}
            continue
        stat, p = ks_2samp(a, b)
        rep[f] = {"ks_stat": float(stat), "p_value": float(p), "drift": bool(p < alpha)}
    return rep

def main(train_path, prod_path, out_path):
    tr = pd.read_csv(train_path)
    pr = pd.read_csv(prod_path)
    rep = ks_test_drift(tr, pr, FEATURES)
    out = {"alpha": 0.05, "features": rep}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote drift report -> {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--prod", required=True)
    ap.add_argument("--out", default=str(REPORTS_DIR / "drift_report.json"))
    args = ap.parse_args()
    main(args.train, args.prod, args.out)
