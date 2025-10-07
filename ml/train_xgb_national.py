import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import os

# ===== CONFIG =====
CSV_PATH = "data/processed/floodzy_train_ready.csv"  # bisa ganti ke floodzy_train_national.csv nanti
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

# ===== MAIN =====
def main():
    X, y = load_data(CSV_PATH)

    print("\n🧮 Distribusi label flood_event:")
    print(y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

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
