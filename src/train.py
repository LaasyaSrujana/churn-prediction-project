# train.py
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import resample
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier

# -------------------------
# Load dataset
# -------------------------
data_path = os.path.join("..", "data", "raw", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
df = pd.read_csv(data_path)

# -------------------------
# Basic cleaning
# -------------------------
df = df.dropna()
df = df.drop_duplicates()

# -------------------------
# Encode categorical columns
# -------------------------
label_encoders = {}
for col in df.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# -------------------------
# Feature/Target split
# -------------------------
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Save feature order for inference
feature_order = list(X.columns)

# -------------------------
# Handle class imbalance
# -------------------------
X = pd.concat([X, y], axis=1)
majority = X[X.Churn == 0]
minority = X[X.Churn == 1]

minority_up = resample(minority, replace=True, n_samples=len(majority), random_state=42)

df_balanced = pd.concat([majority, minority_up])

y = df_balanced["Churn"]
X = df_balanced.drop("Churn", axis=1)

# -------------------------
# Train-test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------
# Scaling
# -------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------
# Models
# -------------------------
xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="logloss",
    random_state=42
)

lgbm = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42
)

# Ensemble model
model = VotingClassifier(
    estimators=[("xgb", xgb), ("lgbm", lgbm)],
    voting="soft"
)

# -------------------------
# Train
# -------------------------
model.fit(X_train_scaled, y_train)

# -------------------------
# Evaluate
# -------------------------
y_pred = model.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# -------------------------
# Save everything
# -------------------------
os.makedirs("../models", exist_ok=True)

joblib.dump(model, "../models/churn_model.joblib")
joblib.dump(label_encoders, "../models/label_encoders.joblib")
joblib.dump(scaler, "../models/scaler.joblib")
joblib.dump(feature_order, "../models/feature_order.joblib")

print(" Model, encoders, scaler, feature order saved successfully.")
