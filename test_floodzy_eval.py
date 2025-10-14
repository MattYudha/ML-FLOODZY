import xgboost as xgb
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load model & data
model = xgb.XGBClassifier()
model.load_model("artifacts/xgb_floodzy_national_v2.json")

df = pd.read_csv("data/processed/floodzy_train_ready.csv").dropna()

# 🔧 Convert object columns to numeric if possible
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = pd.to_numeric(df[col], errors='coerce')

X = df.drop(columns=["date", "flood_event"])
y = df["flood_event"]

# Predict
y_pred = model.predict(X)
y_prob = model.predict_proba(X)[:, 1]

# Metrics
print("✅ Floodzy Evaluation Report")
print("AUC:", roc_auc_score(y, y_prob))
print(classification_report(y, y_pred))

# Confusion Matrix
cm = confusion_matrix(y, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Floodzy")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Feature Importance
xgb.plot_importance(model, max_num_features=10)
plt.title("Top 10 Feature Importance")
plt.show()
