import os
import pandas as pd
import numpy as np
import xgboost as xgb
import cupy as cp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score

# ===== CONFIG =====
CSV_PATH = "data/processed/floodzy_new_train.csv"
MODEL_PATH = "artifacts/xgb_floodzy_national_v2_cuda.json"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ===== GPU CHECK =====
try:
    gpu_name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode("utf-8")
    print(f"🎮 GPU Detected: {gpu_name}")
except Exception as e:
    print("⚠️ CUDA device not detected:", e)
    print("Training will fail if GPU is unavailable.")
    raise SystemExit()

# ===== LOAD DATA =====
def load_data(path):
    print(f"\n📥 Reading CSV: {os.path.abspath(path)}")
    df = pd.read_csv(path)

    print(f"Rows: {len(df)} before cleaning")
    df = df.dropna()
    print(f"Rows: {len(df)} after cleaning")

    # --- KONVERSI STRING KE NUMERIK ---
    for col in df.columns:
        if df[col].dtype == "object":
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            print(f"🔢 Kolom '{col}' dikonversi ke numerik.")

    X = df.drop(columns=["date", "flood_event"], errors="ignore")
    y = df["flood_event"]
    return X, y

# ===== FEATURE ENGINEERING =====
def create_features(X: pd.DataFrame) -> pd.DataFrame:
    X_new = X.copy()

    # Rata-rata curah hujan 3 hari terakhir
    X_new["rain_mm_3d_avg"] = X_new["rain_mm"].rolling(window=3, min_periods=1).mean()

    # Interaksi antara curah hujan & tinggi sungai
    if "rain_mm" in X_new.columns and "river_level_cm" in X_new.columns:
        X_new["rain_x_river_interaction"] = X_new["rain_mm"] * X_new["river_level_cm"]

    X_new = X_new.fillna(0)
    print("✨ Fitur baru dibuat:", ["rain_mm_3d_avg", "rain_x_river_interaction"])
    return X_new

# ===== MAIN TRAINING =====
def main():
    X, y = load_data(CSV_PATH)
    X = create_features(X)

    print("\n🧮 Distribusi label flood_event:")
    print(y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # Class imbalance correction
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"⚖️ scale_pos_weight: {scale_pos_weight:.2f}")

    # Convert ke GPU array
    X_train_gpu = cp.array(X_train.values, dtype=cp.float32)
    X_test_gpu = cp.array(X_test.values, dtype=cp.float32)
    y_train_gpu = cp.array(y_train.values, dtype=cp.float32)
    y_test_gpu = cp.array(y_test.values, dtype=cp.float32)

    # ===== TRAIN =====
    print("\n🚀 Training model sepenuhnya di GPU (CUDA)...")
    model = xgb.XGBClassifier(
        tree_method="gpu_hist",     # pastikan full GPU mode
        predictor="gpu_predictor",
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric="auc"
    )

    model.fit(X_train_gpu.get(), y_train_gpu.get())
    print("✅ Training done (GPU)")

    # ===== EVALUATION =====
    y_pred = model.predict(X_test_gpu.get())
    y_prob = model.predict_proba(X_test_gpu.get())[:, 1]

    auc = roc_auc_score(y_test_gpu.get(), y_prob)
    print(f"\n🔥 AUC: {auc:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test_gpu.get(), y_pred, digits=3))

    # ===== SAVE MODEL =====
    os.makedirs("artifacts", exist_ok=True)
    model.save_model(MODEL_PATH)
    print(f"\n💾 Model GPU disimpan ke: {MODEL_PATH}")

if __name__ == "__main__":
    main()
