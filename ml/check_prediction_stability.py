"""
Proxy concept drift via prediction stability.
Compare recent production predictions (mean/std) against baseline stats from training.
Usage:
  python ml/check_prediction_stability.py --model xgb --prod data/prod_sample.csv --out reports/stability_report_xgb.json
Notes:
  - Baseline file expected: reports/baseline_pred_stats_<model>.json
  - If baseline missing, script exits with status 2.
"""
import argparse, json, os, sys
import pandas as pd
import numpy as np
from ml.inference import predict_xgb, predict_lr, load_feature_order
from ml.ml_config import REPORTS_DIR

def _predict_probs(model: str, df: pd.DataFrame):
    order = load_feature_order()
    probs = []
    for _, row in df[order].iterrows():
        payload = row.to_dict()
        if model == "xgb":
            probs.append(predict_xgb(payload)["probability"])
        elif model == "lr":
            probs.append(predict_lr(payload)["probability"])
        else:
            raise ValueError("model must be 'xgb' or 'lr'")
    return np.asarray(probs, dtype=float)

def main(model, prod_path, out_path, mean_tol=0.15, std_tol=0.20):
    base_file = os.path.join("reports", f"baseline_pred_stats_{model}.json")
    if not os.path.exists(base_file):
        print(f"[WARN] baseline file not found: {base_file}", file=sys.stderr)
        sys.exit(2)
    base = json.loads(open(base_file,"r",encoding="utf-8").read())
    prod = pd.read_csv(prod_path)

    probs = _predict_probs(model, prod)
    cur = {"mean": float(probs.mean()), "std": float(probs.std()), "count": int(probs.size)}

    # Simple relative shift checks (you can replace with z-score/quantile shift later)
    mean_shift = abs(cur["mean"] - base["mean"]) / max(base["mean"], 1e-6)
    std_shift  = abs(cur["std"]  - base["std"])  / max(base["std"],  1e-6)

    breached = (mean_shift > mean_tol) or (std_shift > std_tol)
    report = {"baseline": base, "current": cur, "mean_rel_shift": mean_shift, "std_rel_shift": std_shift, "breached": breached}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote stability report -> {out_path}")
    sys.exit(1 if breached else 0)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["xgb","lr"], required=True)
    ap.add_argument("--prod", required=True)
    ap.add_argument("--out", default=str(REPORTS_DIR / "stability_report.json"))
    ap.add_argument("--mean_tol", type=float, default=0.15)
    ap.add_argument("--std_tol", type=float, default=0.20)
    args = ap.parse_args()
    main(args.model, args.prod, args.out, args.mean_tol, args.std_tol)
