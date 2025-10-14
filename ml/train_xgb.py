# ml/train_xgb.py
from pathlib import Path
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import argparse  # ### TAMBAHKAN INI ###

# HAPUS BARIS INI: CSV_PATH = Path("data/processed/floodzy_train_ready.csv")
LABEL = "flood_event"
USE_GPU = True

NUM_COLS = [
    "rain_mm", "river_level_cm", "tide_cm", "temperature_avg",
    "humidity_avg", "wind_speed", "elevation_m", "api_7d"
]
REQUIRED_COLS = {"date", "region_id", *NUM_COLS, LABEL}


def load_data(path: Path):
    print("Reading CSV:", path.resolve())
    df = pd.read_csv(
        path,
        sep=",",
        encoding="utf-8-sig",
        engine="python",
        skip_blank_lines=True,
        skipinitialspace=True
    )
    df.columns = [c.strip() for c in df.columns]

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise RuntimeError(f"Header/kolom hilang di CSV: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df[LABEL] = pd.to_numeric(df[LABEL], errors="coerce").astype("Int64")

    for c in ["region_id", *NUM_COLS]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["region_id", *NUM_COLS, LABEL])
    df[LABEL] = df[LABEL].astype(int)
    print(f"Rows: {before} -> {len(df)} after cleaning")

    X = df[["region_id", *NUM_COLS]].copy()
    y = df[LABEL].copy()
    return X, y


def build_model(use_gpu: bool) -> XGBClassifier:
    params = dict(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        max_bin=256,
        eval_metric="auc",
        tree_method="hist",
        device="cuda" if use_gpu else "cpu",
        random_state=42,
    )
    return XGBClassifier(**params)


# ### MODIFIKASI: main() sekarang menerima argumen data_path ###
def main(data_path: str):
    X, y = load_data(Path(data_path))

    try:
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    model = build_model(USE_GPU)

    try:
        from xgboost.callback import EarlyStopping
        es = EarlyStopping(rounds=50, save_best=True, metric_name="auc")
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            callbacks=[es],
            verbose=False
        )
        print("✅ Training done with callback EarlyStopping")
    except Exception:
        try:
            model.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)],
                early_stopping_rounds=50,
                verbose=False
            )
            print("✅ Training done with early_stopping_rounds")
        except Exception:
            model.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)],
                verbose=False
            )
            print("✅ Training done without early stopping")

    proba = model.predict_proba(X_valid)[:, 1]
    preds = (proba >= 0.5).astype(int)
    print("AUC:", roc_auc_score(y_valid, proba))
    print(classification_report(y_valid, preds, digits=3))

    out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "xgb_floodzy_model.json"
    model.save_model(out_path.as_posix())
    print(f"✅ Model saved to {out_path}")


# ### TAMBAHKAN BLOK INI untuk memproses argumen dari command line ###
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to the training CSV file")
    args = parser.parse_args()
    main(args.data)