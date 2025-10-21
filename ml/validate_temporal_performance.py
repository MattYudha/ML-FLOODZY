import pandas as pd
import xgboost as xgb
import os
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

MODEL_PATH = "artifacts/xgb_floodzy_national_v2_cuda.json"
OUTPUT_DIR = "reports/temporal_validation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def evaluate_period(dataset_path):
    df = pd.read_csv(dataset_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Pisahkan fitur dan target
    X = df.drop(columns=["flood_event", "date"])
    y = df["flood_event"]

    # Pastikan data numerik
    for col in X.select_dtypes(include="object").columns:
        X[col] = pd.factorize(X[col])[0]

    print(f"🧠 Evaluating on {os.path.basename(dataset_path)} ... Rows: {len(X)}")

    # Load model
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    model.set_params(device="cuda")

    preds = model.predict(X)
    probas = model.predict_proba(X)[:, 1]

    auc = roc_auc_score(y, probas)
    report = classification_report(y, preds, output_dict=True)
    cm = confusion_matrix(y, preds)

    # Simpan Confusion Matrix
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Flood", "Flood"], yticklabels=["No Flood", "Flood"])
    plt.title(f"Confusion Matrix - {os.path.basename(dataset_path)}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"CM_{os.path.basename(dataset_path)}.png"))
    plt.close()

    print(f"✅ AUC: {auc:.3f}")
    print(f"Accuracy: {report['accuracy']:.3f}")
    print(f"F1 Flood: {report['1']['f1-score']:.3f}")
    print(f"F1 NoFlood: {report['0']['f1-score']:.3f}")

    return {"dataset": os.path.basename(dataset_path), "AUC": auc, "Accuracy": report["accuracy"], "F1_Flood": report["1"]["f1-score"], "F1_NoFlood": report["0"]["f1-score"]}

if __name__ == "__main__":
    results = []
    for dataset in ["data/processed/floodzy_eval_2023.csv", "data/processed/floodzy_eval_2025.csv"]:
        results.append(evaluate_period(dataset))

    df_results = pd.DataFrame(results)
    out_path = os.path.join(OUTPUT_DIR, "temporal_results.csv")
    df_results.to_csv(out_path, index=False)
    print("\n📊 Ringkasan hasil disimpan di:", out_path)
