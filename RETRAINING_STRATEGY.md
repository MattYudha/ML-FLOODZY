# Floodzy — Retraining Strategy

## Triggers
- **Performance drop** vs baseline: ROC-AUC ↓ >5 pts, PR-AUC ↓ >5 pts, Brier ↑ >10% (LR/XGB); RMSE ↑ >15% (RF).
- **Data drift**: KS-test (lihat `ml/detect_drift.py`) -> ≥3 fitur drift=True selama ≥2 minggu.
- **Label shift**: proporsi kelas berubah >10 poin.

## Cadence
- Light retrain: tiap 2 minggu (rolling window).
- Full retrain: 1–3 bulan atau pasca event ekstrem.

## Proses
1) Freeze data hingga T.  
2) Train `train_lr.py`, `train_xgb.py`, `train_rf.py`.  
3) Eval + plots (ROC/PR/Calibration).  
4) Acceptance gates vs prod.  
5) Tulis `models/versions.json` (timestamp + optional git SHA).  
6) Deploy artefak; simpan N versi terakhir (rollback-ready).

## Monitoring
- API logs: simpan `model_version`, `latency_ms`, skor risiko (agregasi harian).
- Dashboard drift: render laporan KS-test dari `reports/drift_report.json`.
