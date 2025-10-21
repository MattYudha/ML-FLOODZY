import pandas as pd
import os

# Pastikan path data ada
base_dir = "data/processed"
os.makedirs(base_dir, exist_ok=True)

# Load file utama
df = pd.read_csv(os.path.join(base_dir, "floodzy_new_train.csv"))

# Pastikan kolom date ada
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Filter data
df_2023 = df[df["date"].dt.year <= 2023]
df_2025 = df[(df["date"].dt.year >= 2024) & (df["date"].dt.year <= 2025)]

# Simpan file hasil split
df_2023.to_csv(os.path.join(base_dir, "floodzy_eval_2023.csv"), index=False)
df_2025.to_csv(os.path.join(base_dir, "floodzy_eval_2025.csv"), index=False)

print("✅ File berhasil dibuat:")
print(" - data/processed/floodzy_eval_2023.csv")
print(" - data/processed/floodzy_eval_2025.csv")
