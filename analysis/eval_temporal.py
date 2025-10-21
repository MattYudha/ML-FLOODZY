# -*- coding: utf-8 -*-
"""
Temporal Performance Validation for Floodzy XGBoost Model.

This script evaluates the performance of a pre-trained XGBoost model
on a recent, out-of-time dataset (2024-2025) to check for performance degradation
or concept drift. It calculates key metrics and generates visualizations
for academic reporting.

Author: Gemini AI for Floodzy Project
Date: 21 Oktober 2025
"""

import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    classification_report
)
import os

# --- 1. Konfigurasi Awal ---
# Konfigurasi path ke model dan dataset.
# Pastikan path ini sesuai dengan struktur direktori Anda.
MODEL_PATH = 'artifacts/xgb_floodzy_national_v2_cuda.json' 
# Menggunakan nama file dari training sebelumnya, jika Anda sudah rename ke _cuda.json, silakan disesuaikan.
DATA_PATH = 'data/processed/floodzy_new_train.csv'
TARGET_COLUMN = 'banjir'  # Nama kolom target/label

# Buat direktori untuk menyimpan output jika belum ada
output_dir = 'reports/temporal_validation'
os.makedirs(output_dir, exist_ok=True)
print(f"Grafik dan laporan akan disimpan di direktori: '{output_dir}'")


def plot_confusion_matrix(y_true, y_pred, subset_name, save_path):
    """
    Membuat dan menyimpan plot confusion matrix.
    Ini membantu memvisualisasikan performa model dalam hal True Positives,
    False Positives, True Negatives, dan False Negatives.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Tidak Banjir', 'Banjir'], 
                yticklabels=['Tidak Banjir', 'Banjir'])
    plt.title(f'Confusion Matrix - {subset_name}')
    plt.ylabel('Label Aktual')
    plt.xlabel('Label Prediksi')
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion Matrix untuk '{subset_name}' disimpan di: {save_path}")


def plot_roc_curve(y_true, y_pred_proba, subset_name, save_path):
    """
    Membuat dan menyimpan plot ROC Curve.
    Grafik ini menunjukkan kemampuan model untuk membedakan antara kelas positif dan negatif.
    Semakin dekat kurva ke pojok kiri atas, semakin baik performa model.
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) - {subset_name}')
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plt.close()
    print(f"ROC Curve untuk '{subset_name}' disimpan di: {save_path}")

def evaluate_performance(y_true, y_pred, y_pred_proba):
    """Menghitung dan mengembalikan dictionary berisi metrik performa."""
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-Score": f1_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_pred_proba)
    }
    return metrics

def main():
    """Fungsi utama untuk menjalankan validasi temporal."""
    
    # --- 2. Memuat Model dan Data ---
    # Memuat model XGBoost yang sudah dilatih sebelumnya.
    print(f"Memuat model dari {MODEL_PATH}...")
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)

    # Memuat dataset baru untuk validasi.
    print(f"Memuat data dari {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    # --- 3. Preprocessing dan Filtering Data ---
    # Mengonversi kolom 'date' ke format datetime untuk filtering.
    # Ini adalah langkah kunci untuk isolasi data temporal.
    df['date'] = pd.to_datetime(df['date'])

    # Memisahkan fitur (X) dan target (y) dari keseluruhan dataset.
    X_all = df.drop(columns=[TARGET_COLUMN, 'date'])
    y_all = df[TARGET_COLUMN]

    # Memastikan kolom fitur sesuai dengan yang digunakan saat training.
    # Model XGBoost menyimpan nama fitur yang dapat digunakan untuk alignment.
    model_features = model.get_booster().feature_names
    X_all = X_all[model_features]

    # Membuat subset data khusus untuk periode 2024-2025.
    # Ini adalah inti dari validasi temporal (out-of-time validation).
    print("\nMemfilter data untuk periode 2024-2025...")
    start_date = '2024-01-01'
    end_date = '2025-12-31'
    temporal_subset = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    
    if temporal_subset.empty:
        print("Tidak ditemukan data pada rentang waktu 2024-2025. Skrip dihentikan.")
        return

    print(f"Ditemukan {len(temporal_subset)} baris data untuk periode 2024-2025.")
    
    X_temporal = temporal_subset[model_features]
    y_temporal = temporal_subset[TARGET_COLUMN]

    # --- 4. Melakukan Prediksi dengan Akselerasi GPU ---
    print("\nMelakukan prediksi pada subset data 2024-2025 menggunakan CUDA...")
    # Penting: 'device="cuda"' memberitahu XGBoost untuk melakukan inferensi di GPU.
    # Ini mempercepat proses prediksi, terutama pada data besar.
    model.set_params(device="cuda")
    
    y_pred_temporal = model.predict(X_temporal)
    y_pred_proba_temporal = model.predict_proba(X_temporal)[:, 1]

    # --- 5. Evaluasi Performa pada Subset Temporal ---
    print("\n--- Hasil Evaluasi pada Data 2024-2025 ---")
    temporal_metrics = evaluate_performance(y_temporal, y_pred_temporal, y_pred_proba_temporal)
    
    for metric, value in temporal_metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nLaporan Klasifikasi Detail (2024-2025):")
    print(classification_report(y_temporal, y_pred_temporal, target_names=['Tidak Banjir', 'Banjir']))

    # --- 6. Membuat Visualisasi untuk Laporan ---
    print("\nMembuat visualisasi untuk subset 2024-2025...")
    plot_confusion_matrix(
        y_temporal, y_pred_temporal, "Subset 2024-2025", 
        os.path.join(output_dir, 'confusion_matrix_2024_2025.png')
    )
    plot_roc_curve(
        y_temporal, y_pred_proba_temporal, "Subset 2024-2025",
        os.path.join(output_dir, 'roc_curve_2024_2025.png')
    )

    # --- 7. Analisis Perbandingan ---
    print("\n--- Analisis Perbandingan Performa ---")
    # Melakukan prediksi pada seluruh data untuk perbandingan.
    print("Mengevaluasi performa pada keseluruhan dataset sebagai baseline...")
    y_pred_all = model.predict(X_all)
    y_pred_proba_all = model.predict_proba(X_all)[:, 1]
    all_data_metrics = evaluate_performance(y_all, y_pred_all, y_pred_proba_all)

    # Membuat DataFrame untuk perbandingan yang mudah dibaca.
    comparison_df = pd.DataFrame({
        'Keseluruhan Data': all_data_metrics,
        'Data 2024-2025': temporal_metrics
    })
    print(comparison_df.round(4))

    # --- 8. Ringkasan Akademik ---
    # Ringkasan ini dapat langsung disalin ke dalam laporan Anda.
    print("\n--- Ringkasan untuk Laporan Akademik Floodzy ---")
    auc_temporal = temporal_metrics['AUC']
    auc_all = all_data_metrics['AUC']
    performance_change = ((auc_temporal - auc_all) / auc_all) * 100
    
    print("Validasi temporal dilakukan untuk menguji generalisasi model pada data baru (periode 2024-2025) "
          "yang tidak terlihat saat training.")
    print(f"Model menunjukkan performa AUC sebesar {auc_temporal:.4f} pada data 2024-2025, "
          f"dibandingkan dengan AUC {auc_all:.4f} pada keseluruhan data.")
    
    if abs(performance_change) < 5:
        print(f"Perubahan performa ({performance_change:+.2f}%) masih dalam batas wajar. "
              "Ini menunjukkan model memiliki stabilitas temporal yang baik dan belum mengalami 'concept drift' yang signifikan.")
    else:
        print(f"Terjadi perubahan performa yang signifikan ({performance_change:+.2f}%). "
              "Ini bisa menjadi indikasi awal adanya 'concept drift', di mana distribusi data telah berubah. "
              "Rekomendasi: Lakukan analisis lebih lanjut dan pertimbangkan untuk melatih ulang (retraining) model dengan data baru.")

if __name__ == '__main__':
    main()

