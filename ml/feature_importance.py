from ml.ml_config import MODELS_DIR, FEATURE_LIST_JSON
import os, json, joblib, logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
LOGGER = logging.getLogger(__name__)

def feature_importance():
    path = os.path.join(MODELS_DIR, "rf_pipeline.joblib")
    pipe = joblib.load(path)
    model = getattr(pipe, "named_steps", {}).get("rf", None)
    if model is None or not hasattr(model, "feature_importances_"):
        raise RuntimeError("RF model or feature_importances_ not available.")
    with open(os.path.join(MODELS_DIR, FEATURE_LIST_JSON), "r", encoding="utf-8") as f:
        order = json.load(f)
    imps = model.feature_importances_
    data = sorted([{"feature": f, "importance_score": float(i)} for f, i in zip(order, imps)], key=lambda x: x["importance_score"], reverse=True)
    return data
