import xgboost as xgb, json

print(json.dumps(xgb.build_info(), indent=2)[:300])  # harus ada "USE_CUDA"

m = xgb.XGBClassifier(tree_method="hist", device="cuda")
print("✅ XGBoost CUDA OK")
