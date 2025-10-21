# -*- coding: utf-8 -*-
"""
Temporal Performance Report Generator for Floodzy

- Load trained XGBoost model
- Evaluate on 2023 and 2025 evaluation datasets
- Compute metrics, draw ROC & Confusion Matrix
- Generate Markdown and PDF report with embedded images

Requirements:
  pip install reportlab matplotlib pandas xgboost scikit-learn

Author: Floodzy
"""

import os
import io
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors

import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)

# ---------------- CONFIG ----------------
MODEL_PATH = "artifacts/xgb_floodzy_national_v2_cuda.json"
DATA_DIR = "data/processed"
EVAL_FILES = {
    "2019–2023": os.path.join(DATA_DIR, "floodzy_eval_2023.csv"),
    "2024–2025": os.path.join(DATA_DIR, "floodzy_eval_2025.csv"),
}
TARGET_COL = "flood_event"  # kolom label biner {0,1}

REPORT_DIR = "reports/temporal_validation"
os.makedirs(REPORT_DIR, exist_ok=True)

MD_PATH = os.path.join(REPORT_DIR, "Temporal_Performance_Report.md")
PDF_PATH = os.path.join(REPORT_DIR, "Temporal_Performance_Report.pdf")


# -------------- UTIL: PLOTTING --------------
def plot_confusion_matrix(y_true, y_pred, title, save_path):
    cmatrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig = plt.figure(figsize=(5, 4), dpi=150)
    ax = plt.gca()
    im = ax.imshow(cmatrix, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["NoFlood (0)", "Flood (1)"])
    ax.set_yticklabels(["NoFlood (0)", "Flood (1)"])

    # write numbers
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cmatrix[i, j], ha="center", va="center", color="white" if cmatrix[i, j] > cmatrix.max()/2 else "black")

    fig.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_roc(y_true, y_prob, title, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig = plt.figure(figsize=(5, 4), dpi=150)
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", lw=1)
    plt.xlim([0, 1])
    plt.ylim([0, 1.02])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


# -------------- DATA PREP --------------
def ensure_numeric_features(X: pd.DataFrame) -> pd.DataFrame:
    """Pastikan semua kolom fitur numerik. Jika ada object/category, factorize."""
    non_num = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if non_num:
        for col in non_num:
            X[col] = pd.factorize(X[col])[0]
    return X


def ensure_engineered_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Pastikan fitur turunan konsisten dengan training:
      - rain_mm_3d_avg
      - rain_x_river_interaction
    Jika belum ada, buat sederhana (rolling berdasarkan urutan baris).
    """
    X = X.copy()
    if "rain_mm_3d_avg" not in X.columns and "rain_mm" in X.columns:
        # rolling per region jika ada region_id
        if "region_id" in X.columns:
            X["rain_mm_3d_avg"] = (
                X.groupby("region_id")["rain_mm"].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)
            )
        else:
            X["rain_mm_3d_avg"] = X["rain_mm"].rolling(window=3, min_periods=1).mean()
            X["rain_mm_3d_avg"] = X["rain_mm_3d_avg"].fillna(method="bfill").fillna(0)

    if "rain_x_river_interaction" not in X.columns and {"rain_mm", "river_level_cm"}.issubset(X.columns):
        X["rain_x_river_interaction"] = X["rain_mm"] * X["river_level_cm"]

    return X.fillna(0)


# -------------- EVAL --------------
def eval_one_period(name: str, csv_path: str, model: xgb.XGBClassifier):
    print(f"🧠 Evaluating {name} ...")
    df = pd.read_csv(csv_path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    assert TARGET_COL in df.columns, f"Kolom target '{TARGET_COL}' tidak ditemukan pada {csv_path}"
    y = df[TARGET_COL].astype(int).values

    # drop non-feature columns
    drop_cols = [TARGET_COL]
    if "date" in df.columns:
        drop_cols.append("date")
    X = df.drop(columns=drop_cols, errors="ignore")

    # numeric + engineered features
    X = ensure_numeric_features(X)
    X = ensure_engineered_features(X)

    # Agar urutan kolom match ke booster (kalau tersimpan)
    booster = model.get_booster()
    feat_names = booster.feature_names
    if feat_names and all(f in X.columns for f in feat_names):
        X = X[feat_names]

    # Hindari warning "mismatched devices": gunakan booster.predict(DMatrix(X))
    dtest = xgb.DMatrix(X)
    y_prob = booster.predict(dtest)
    # Jika output prob shape (n,2), ambil kelas 1
    if y_prob.ndim == 2 and y_prob.shape[1] == 2:
        y_prob = y_prob[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    # metrics
    metrics = {
        "rows": len(df),
        "accuracy": accuracy_score(y, y_pred),
        "precision_flood": precision_score(y, y_pred, pos_label=1, zero_division=0),
        "recall_flood": recall_score(y, y_pred, pos_label=1, zero_division=0),
        "f1_flood": f1_score(y, y_pred, pos_label=1, zero_division=0),
        "precision_noflood": precision_score(y, y_pred, pos_label=0, zero_division=0),
        "recall_noflood": recall_score(y, y_pred, pos_label=0, zero_division=0),
        "f1_noflood": f1_score(y, y_pred, pos_label=0, zero_division=0),
        "auc": roc_auc_score(y, y_prob),
    }

    # plots
    roc_path = os.path.join(REPORT_DIR, f"{name.replace(' ', '_')}_roc.png")
    cm_path = os.path.join(REPORT_DIR, f"{name.replace(' ', '_')}_cm.png")
    plot_roc(y, y_prob, f"ROC Curve ({name})", roc_path)
    plot_confusion_matrix(y, y_pred, f"Confusion Matrix ({name})", cm_path)

    return metrics, roc_path, cm_path


# -------------- REPORT (MD + PDF) --------------
def write_markdown(summary_rows, images):
    lines = []
    lines.append(f"# Temporal Performance Report — Floodzy\n")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    lines.append(f"**Model:** `{os.path.basename(MODEL_PATH)}`\n")
    lines.append(f"\n---\n")
    lines.append("## Ringkasan Metrik\n")
    df = pd.DataFrame(summary_rows).set_index("Period")
    lines.append(df.to_markdown(floatfmt=".3f"))
    lines.append("\n")

    for period, (roc_path, cm_path) in images.items():
        lines.append(f"## {period}\n")
        lines.append(f"**ROC Curve**  \n")
        lines.append(f"![ROC]({os.path.relpath(roc_path, start=os.path.dirname(MD_PATH))})\n")
        lines.append(f"\n**Confusion Matrix**  \n")
        lines.append(f"![CM]({os.path.relpath(cm_path, start=os.path.dirname(MD_PATH))})\n")

    lines.append("\n---\n")
    lines.append("### Catatan\n")
    lines.append("- Evaluasi dilakukan dengan memisahkan data ke dua horizon waktu: 2019–2023 (baseline historis) dan 2024–2025 (out-of-time).")
    lines.append("- AUC ≥ 0.95 dan stabil antar-periode menunjukkan robustnes model terhadap perubahan waktu.")
    lines.append("- Jika performa periode terbaru turun ≥5%, lakukan retraining dengan data terbaru.\n")

    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def draw_table_pdf(c, df, x, y, row_height=14, col_padding=8):
    # simple table drawing
    c.setFont("Helvetica-Bold", 11)
    cols = list(df.columns)
    col_widths = []
    # estimate widths
    for col in cols:
        w = max(c.stringWidth(col, "Helvetica-Bold", 11), 80)
        for val in df[col]:
            w = max(w, c.stringWidth(f"{val}", "Helvetica", 10) + 10)
        col_widths.append(w)

    # header
    cx = x
    c.setFillColor(colors.black)
    for i, col in enumerate(cols):
        c.drawString(cx + 2, y, col)
        cx += col_widths[i] + col_padding
    y -= row_height

    # rows
    c.setFont("Helvetica", 10)
    for idx, row in df.iterrows():
        cx = x
        c.drawString(cx + 2, y, str(idx))
        cx += col_widths[0] + col_padding
        for i, col in enumerate(cols[1:], start=1):
            c.drawString(cx + 2, y, f"{row[col]:.3f}" if isinstance(row[col], (int, float, np.floating)) else str(row[col]))
            cx += col_widths[i] + col_padding
        y -= row_height
    return y


def write_pdf(summary_rows, images):
    c = canvas.Canvas(PDF_PATH, pagesize=A4)
    W, H = A4

    # title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(2*cm, H - 2.5*cm, "Temporal Performance Report — Floodzy")
    c.setFont("Helvetica", 10)
    c.drawString(2*cm, H - 3.1*cm, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    c.drawString(2*cm, H - 3.6*cm, f"Model: {os.path.basename(MODEL_PATH)}")

    # table
    df = pd.DataFrame(summary_rows).set_index("Period")
    y = H - 4.6*cm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2*cm, y, "Ringkasan Metrik")
    y -= 0.6*cm

    y = draw_table_pdf(c, df, x=2*cm, y=y)

    c.showPage()

    # images per period
    for period, (roc_path, cm_path) in images.items():
        c.setFont("Helvetica-Bold", 14)
        c.drawString(2*cm, H - 2.5*cm, f"{period}")

        # Embed ROC
        y_img = H - 3.3*cm
        for title, img_path in [("ROC Curve", roc_path), ("Confusion Matrix", cm_path)]:
            if os.path.exists(img_path):
                c.setFont("Helvetica", 11)
                c.drawString(2*cm, y_img, title)
                y_img -= 0.5*cm
                with open(img_path, "rb") as f:
                    img = ImageReader(io.BytesIO(f.read()))
                c.drawImage(img, 2*cm, y_img - 7.5*cm, width=12*cm, height=7.5*cm, preserveAspectRatio=True, anchor="nw")
                y_img -= 8.2*cm
            else:
                c.setFont("Helvetica", 11)
                c.drawString(2*cm, y_img, f"{title}: (gambar tidak ditemukan)")
                y_img -= 0.8*cm

        c.showPage()

    # notes
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2*cm, H - 2.5*cm, "Catatan")
    c.setFont("Helvetica", 10)
    text = c.beginText(2*cm, H - 3.2*cm)
    text.textLines(
        "- Evaluasi memisahkan data ke dua horizon waktu: 2019–2023 (baseline) dan 2024–2025 (out-of-time).\n"
        "- AUC tinggi dan stabil menandakan robustnes terhadap perubahan distribusi waktu.\n"
        "- Jika performa periode terbaru menurun signifikan (≥5%), pertimbangkan retraining model dengan data terbaru."
    )
    c.drawText(text)

    c.save()


# -------------- MAIN --------------
def main():
    print(f"📄 Generate report to: {REPORT_DIR}")
    # load model
    print(f"📦 Loading model: {MODEL_PATH}")
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    # gunakan device CUDA untuk training/predict booster; untuk predict kita pakai DMatrix -> booster.predict
    try:
        model.set_params(device="cuda")
    except Exception:
        pass

    summary_rows = []
    images = {}

    for period, path in EVAL_FILES.items():
        if not os.path.exists(path):
            print(f"⚠️ File tidak ditemukan: {path}, lewati {period}")
            continue
        metrics, roc_path, cm_path = eval_one_period(period, path, model)
        row = {
            "Period": period,
            "Rows": metrics["rows"],
            "AUC": metrics["auc"],
            "Accuracy": metrics["accuracy"],
            "F1_Flood": metrics["f1_flood"],
            "F1_NoFlood": metrics["f1_noflood"],
            "Precision_Flood": metrics["precision_flood"],
            "Recall_Flood": metrics["recall_flood"],
        }
        summary_rows.append(row)
        images[period] = (roc_path, cm_path)

    # Markdown + PDF
    write_markdown(summary_rows, images)
    write_pdf(summary_rows, images)

    print(f"✅ Markdown saved: {MD_PATH}")
    print(f"✅ PDF saved: {PDF_PATH}")


if __name__ == "__main__":
    main()
