# How this patch works
The training scripts (`train_lr.py`, `train_xgb.py`, `train_rf.py`, `train_rf_grid.py`) should:
- Write `models/versions.json` with `{tag, git_sha}` per model
- For LR/XGB, also write `reports/baseline_pred_stats_<model>.json` summarizing predicted probability distribution on the test split

If you prefer manual patching, search for:
  - line containing: `metrics = {...}`
  - Right before the next `LOGGER.info(...)`, insert:
    - ` _write_pred_baseline('<model>', y_prob)` (for LR/XGB)
    - ` _write_versions('<model>', tag)`
