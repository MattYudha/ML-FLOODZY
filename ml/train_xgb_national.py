import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import os
import numpy as np

# ===== CONFIG =====
CSV_PATH = "data/processed/floodzy_train_ready.csv"
MODEL_PATH = "artifacts/xgb_floodzy_national_v2.json"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ===== LOAD DATA =====
def load_data(path):
    print(f"📥 Reading CSV: {os.path.abspath(path)}")
    df = pd.read_csv(path)

    print(f"Rows: {len(df)} before cleaning")
    df = df.dropna()
    print(f"Rows: {len(df)} after cleaning")

    X = df.drop(columns=["date", "flood_event"])
    y = df["flood_event"]
    return X, y

## --- PENAMBAHAN: Fungsi untuk Feature Engineering ---
def create_features(X: pd.DataFrame) -> pd.DataFrame:
    """Membuat fitur-fitur baru dari data yang ada."""
    X_new = X.copy()
    
    # Fitur turunan waktu (contoh: rata-rata curah hujan 3 hari terakhir)
    # Note: Ini hanya contoh sederhana, idealnya dilakukan pada data time-series utuh
    X_new['rain_mm_3d_avg'] = X_new['rain_mm'].rolling(window=3, min_periods=1).mean()
    
    # Fitur interaksi
    X_new['rain_x_river_interaction'] = X_new['rain_mm'] * X_new['river_level_cm']
    
    # Isi nilai NaN yang mungkin muncul dari rolling window
    X_new = X_new.fillna(0)
    
    print("✨ Fitur baru dibuat:", ['rain_mm_3d_avg', 'rain_x_river_interaction'])
    return X_new

# ===== MAIN =====
def main():
    X, y = load_data(CSV_PATH)

    ## --- PENAMBAHAN: Memanggil fungsi feature engineering ---
    X = create_features(X)

    print("\n🧮 Distribusi label flood_event:")
    print(y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    ## --- PENAMBAHAN: Menghitung scale_pos_weight untuk class imbalance ---
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"⚖️ Menggunakan scale_pos_weight: {scale_pos_weight:.2f}")

    # ===== TRAIN =====
    print("\n🚀 Training model on GPU (CUDA)...")
    model = xgb.XGBClassifier(
        tree_method="hist",
        device="cuda",
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        scale_pos_weight=scale_pos_weight  # ## MODIFIKASI: Menambahkan parameter ini
    )

    model.fit(X_train, y_train)
    print("✅ Training done")

    # ===== EVALUATION =====
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    try:
        auc = roc_auc_score(y_test, y_prob)
    except ValueError:
        auc = float("nan")

    print(f"\nAUC: {auc:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=3))

    # ===== SAVE MODEL =====
    os.makedirs("artifacts", exist_ok=True)
    model.save_model(MODEL_PATH)
    print(f"💾 Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()