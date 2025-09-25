Copy these files over your v2 tree (same paths), then:
1) Commit and push. CI will generate hashed requirements.txt from requirements.in and enforce --require-hashes.
2) Rebuild Docker image (it uses the compiled requirements.txt).
3) Retrain so that `models/versions.json` is written by training jobs (ensure training scripts updated accordingly).

Updated/Added:
- ml/ml_config.py  (adds DOMAIN_BOUNDS constants + docs)
- ml/schemas.py    (reads bounds from DOMAIN_BOUNDS)
- ml/inference.py  (returns model_version from models/versions.json)
- ml/detect_drift.py  (KS-test drift detector)
- tests/test_interactions.py (logical interaction tests)
- .github/workflows/ci.yml (pip-tools + hashes)
- RETRAINING_STRATEGY.md
