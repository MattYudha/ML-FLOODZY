# ml/inference.py

from __future__ import annotations
import random # Impor random untuk membuat data palsu

# PERINGATAN: File ini dimodifikasi untuk PENGUJIAN KONEKSI SAJA.
# Ini tidak memanggil model ML yang sebenarnya.

def predict_lr(data: dict, threshold: float = 0.5):
    # Tidak diimplementasikan untuk pengujian ini
    return {}

def predict_rf(data: dict):
    # Tidak diimplementasikan untuk pengujian ini
    return {}

def predict_xgb(data: dict, threshold: float = 0.5):
    """
    Fungsi ini mem-bypass model asli dan mengembalikan
    prediksi palsu (dummy) untuk tujuan pengujian.
    """
    print(f"Menerima data untuk prediksi (dummy): {data}")
    
    # Membuat probabilitas acak antara 0.6 dan 0.95
    dummy_probability = random.uniform(0.6, 0.95)
    dummy_label = 1 if dummy_probability >= threshold else 0
    
    risk_label = "LOW"
    if dummy_probability > 0.33:
        risk_label = "MED"
    if dummy_probability > 0.66:
        risk_label = "HIGH"
        
    # Mengembalikan output yang cocok dengan skema FloodPotentialOut
    return {
        "probability": dummy_probability,
        "label": dummy_label,
        "features_order": ["latitude", "longitude", "water_level"]
    }