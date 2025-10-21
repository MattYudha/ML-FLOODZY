# -*- coding: utf-8 -*-
"""
Feature Importance Analysis for Floodzy XGBoost Model.

Analisis ini menampilkan dua pendekatan utama:
1. XGBoost built-in importance (berdasarkan Gain)
2. SHAP (SHapley Additive exPlanations) untuk interpretasi mendalam

File ini aman untuk dijalankan di CPU atau GPU, dan hasilnya siap digunakan
untuk laporan akademik Floodzy.

Author: Gemini AI for Floodzy Project
Date: 21 Oktober 2025
"""

import os
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

# === 1. KONFIGURASI DASAR ===
MODEL_PATH = "artifacts/xgb_floodzy_national_v2_cuda.json"
DATA_PATH = "data/processed/floodzy_new_train.csv"
TARGET_COLUMN = "banjir"

OUTPUT_DIR = "reports/feature_importance"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"📊 Grafik dan laporan akan disimpan di: '{OUTPUT_DIR}'")


def main():
    # === 2. LOAD MODEL & DATA ===
    print(f"\n📦 Memuat model dari: {MODEL_PATH}")
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)

    try:
        model.set_params(device="cuda")
        print("🎮 Model dikonfigurasi untuk GPU (CUDA).")
    except Exception:
        print("⚠️ CUDA tidak tersedia, model berjalan di CPU.")

    print(f"\n📄 Memuat data dari: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    # === 3. PERSIAPAN DATA ===
    cols_to_drop = [col for col in [TARGET_COLUMN, "date"] if col in df.columns]
    X = df.drop(columns=cols_to_drop)

    # Konversi kolom non-numerik
    non_numeric_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if non_numeric_cols:
        print(f"🔄 Mengonversi kolom non-numerik ke numerik: {non_numeric_cols}")
        for col in non_numeric_cols:
            X[col] = pd.factorize(X[col])[0]
        print("✅ Konversi selesai.")

    model_features = model.get_booster().feature_names
    if model_features:
        print("🔧 Menyesuaikan urutan kolom berdasarkan feature_names...")
        model_features_clean = [f for f in model_features if f is not None]
        missing_features = set(model_features_clean) - set(X.columns)
        if missing_features:
            raise ValueError(f"Kolom berikut tidak ditemukan di dataset: {missing_features}")
        X = X[model_features_clean]
    else:
        print("⚠️ Model tidak menyimpan feature_names, menggunakan urutan kolom CSV.")

    # === 4. XGBOOST BUILT-IN IMPORTANCE ===
    print("\n📈 Membuat XGBoost feature importance (Top 10)...")
    fig, ax = plt.subplots(figsize=(10, 8))
    xgb.plot_importance(model, max_num_features=10, height=0.8, ax=ax, importance_type="gain", show_values=False)
    ax.set_title("Top 10 Fitur Paling Berpengaruh (Berdasarkan Gain)", fontsize=14)
    ax.set_xlabel("Kontribusi Relatif (Gain)")
    ax.set_ylabel("Fitur")
    plt.tight_layout()

    path_importance = os.path.join(OUTPUT_DIR, "xgb_feature_importance_top10.png")
    plt.savefig(path_importance)
    plt.close()
    print(f"💾 Disimpan: {path_importance}")

    # === 5. SHAP FEATURE IMPORTANCE ===
    print("\n🔍 Menghitung SHAP values untuk interpretasi model...")
    try:
        print("🚀 Mencoba GPUTreeExplainer (GPU mode)...")
        explainer = shap.GPUTreeExplainer(model, X)
        shap_values = explainer(X, check_additivity=False)
        print("✅ Menggunakan GPUTreeExplainer.")
    except Exception as e:
        print(f"⚠️ Gagal menggunakan GPU SHAP: {e}")
        print("🔁 Beralih ke TreeExplainer (CPU mode)...")
        explainer = shap.TreeExplainer(model, X)
        shap_values = explainer(X, check_additivity=False)
        print("✅ Menggunakan TreeExplainer CPU.")

    print("\n📊 Membuat SHAP Summary Plot...")
    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.title("SHAP Summary Plot: Pengaruh Fitur terhadap Prediksi Banjir", fontsize=14)
    plt.tight_layout()

    shap_path = os.path.join(OUTPUT_DIR, "shap_summary_plot.png")
    plt.savefig(shap_path)
    plt.close()
    print(f"💾 Disimpan: {shap_path}")

    # === 6. RINGKASAN NARATIF ===
    feature_names = X.columns
    shap_values_array = shap_values.values if hasattr(shap_values, "values") else shap_values
    if len(shap_values_array.shape) == 3:  # handle multiclass
        shap_values_array = shap_values_array[:, :, 1]

    shap_sum = abs(shap_values_array).mean(axis=0)
    df_shap = pd.DataFrame({"feature": feature_names, "importance": shap_sum}).sort_values("importance", ascending=False)
    top_feature = df_shap.iloc[0]["feature"]

    print("\n📘 --- RINGKASAN LAPORAN AKADEMIK FLOODZY ---")
    print("Analisis feature importance dilakukan untuk memahami faktor-faktor utama yang memengaruhi prediksi banjir.")
    print("\nMetode 1️⃣: XGBoost Feature Importance (Gain)")
    print("Menunjukkan kontribusi relatif fitur dalam menurunkan kesalahan model (loss).")
    print("\nMetode 2️⃣: SHAP (SHapley Additive Explanations)")
    print("Memberikan wawasan arah pengaruh fitur terhadap hasil prediksi (positif atau negatif).")
    print(f"\n🎯 Kesimpulan: Fitur '{top_feature}' memiliki pengaruh terbesar terhadap prediksi kemungkinan terjadinya banjir.")

    # === 7. EXPORT HASIL ===
    df_shap.to_csv(os.path.join(OUTPUT_DIR, "shap_feature_importance_values.csv"), index=False)
    print(f"📄 Data penting SHAP disimpan di: {OUTPUT_DIR}/shap_feature_importance_values.csv")

    print("\n✅ Analisis feature importance selesai.")


if __name__ == "__main__":
    main()
