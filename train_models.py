import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score

# 1. Dataset loading (Working GitHub mirror for Cleveland Heart Disease)
url = "https://raw.githubusercontent.com/mrdbourke/zero-to-mastery-ml/master/data/heart-disease.csv"

try:
    print("Downloading Cleveland Heart Disease dataset...")
    df = pd.read_csv(url)
except Exception:
    # Backup source agar network block kare
    backup_url = "https://raw.githubusercontent.com/ageron/data/main/heart/heart.csv"
    df = pd.read_csv(backup_url)

print(f"Dataset successfully loaded! Shape: {df.shape}")

X = df.drop(columns=["target"])
y = df["target"]

# 2. Train-Test Split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Scaling continuous features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Models initialize
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.05, eval_metric="logloss", random_state=42)
}

results = {}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n--- Model Training & Cross-Validation Comparison ---")
for name, model in models.items():
    if name == "Logistic Regression":
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=skf, scoring="roc_auc")
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        probs = model.predict_proba(X_test_scaled)[:, 1]
    else:
        cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="roc_auc")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    rec = recall_score(y_test, preds)
    roc = roc_auc_score(y_test, probs)

    results[name] = {"model": model, "roc": roc, "accuracy": acc, "recall": rec}
    print(f"\n{name}:")
    print(f"  5-Fold CV ROC-AUC: {cv_scores.mean():.4f}")
    print(f"  Test Accuracy:    {acc:.4f}")
    print(f"  Test Recall:      {rec:.4f}")
    print(f"  Test ROC-AUC:     {roc:.4f}")

# 5. Best model save (Random Forest)
best_model = results["Random Forest"]["model"]

joblib.dump(best_model, "heart_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(list(X.columns), "feature_columns.pkl")
print("\n[SUCCESS] Model, Scaler, and Features saved successfully!")