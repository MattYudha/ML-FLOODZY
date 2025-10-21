import pandas as pd
import os

base_dir = "data/processed"

for fname in ["floodzy_eval_2023.csv", "floodzy_eval_2025.csv"]:
    path = os.path.join(base_dir, fname)
    df = pd.read_csv(path)

    print(f"🔧 Memproses {fname}...")

    # Tambahkan kolom baru sesuai formula training
    if "rain_mm" in df.columns:
        df["rain_mm_3d_avg"] = df["rain_mm"].rolling(window=3, min_periods=1).mean()
    else:
        df["rain_mm_3d_avg"] = 0

    if "rain_mm" in df.columns and "river_level_cm" in df.columns:
        df["rain_x_river_interaction"] = df["rain_mm"] * df["river_level_cm"]
    else:
        df["rain_x_river_interaction"] = 0

    # Simpan file hasil update
    df.to_csv(path, index=False)
    print(f"✅ {fname} diperbarui dengan fitur tambahan.")

print("\nSemua dataset evaluasi telah diperbarui.")
