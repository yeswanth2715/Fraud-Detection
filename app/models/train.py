import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from xgboost import XGBClassifier

from app.services.feature_engineering import create_features


# --------------------------------------------------
# Config
# --------------------------------------------------

DATA_PATH = "data/User0_credit_card_transactions.csv"
MODEL_DIR = "models"
METRICS_DIR = "metrics"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)


# --------------------------------------------------
# Load Data
# --------------------------------------------------

def load_data():
    df = pd.read_csv(DATA_PATH)

    df["Is Fraud?"] = df["Is Fraud?"].map({"Yes": 1, "No": 0})

    return df


# --------------------------------------------------
# Automatic Feature Split
# --------------------------------------------------

def split_features_target(df, target_column="Is Fraud?"):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    numeric_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    return X, y, numeric_features, categorical_features


# --------------------------------------------------
# F1-Optimized Threshold
# --------------------------------------------------

def find_best_f1_threshold(y_true, y_proba):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)

    best_index = np.argmax(f1_scores)

    if best_index >= len(thresholds):
        return 0.5

    return thresholds[best_index]


# --------------------------------------------------
# Main Training Pipeline
# --------------------------------------------------

def main():

    # 1️⃣ Load + Feature Engineering
    df = load_data()
    df = create_features(df)

    print("Total Rows:", len(df))
    print("Fraud Ratio:", df["Is Fraud?"].mean())

    X, y, numeric_features, categorical_features = split_features_target(df)

    # 2️⃣ Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 3️⃣ Handle Class Imbalance
    fraud_ratio = (len(y_train) - sum(y_train)) / max(sum(y_train), 1)

    # --------------------------------------------------
    # Preprocessing Pipelines
    # --------------------------------------------------

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features)
        ]
    )

    # --------------------------------------------------
    # Model (Goldilocks XGBoost)
    # --------------------------------------------------

    model = XGBClassifier(
        eval_metric="logloss",
        random_state=42,
        scale_pos_weight=fraud_ratio,
        n_estimators=800,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    # --------------------------------------------------
    # Cross-Validation (PR-AUC Focused)
    # --------------------------------------------------

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    pr_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        scoring="average_precision",
        cv=cv,
        n_jobs=-1
    )

    print("Cross-validated PR-AUC:", pr_scores.mean())

    # --------------------------------------------------
    # Train Final Model
    # --------------------------------------------------

    pipeline.fit(X_train, y_train)

    y_proba = pipeline.predict_proba(X_test)[:, 1]

    best_threshold = find_best_f1_threshold(y_test, y_proba)
    y_pred = (y_proba >= best_threshold).astype(int)

    results = {
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "best_threshold": float(best_threshold)
    }

    print(json.dumps(results, indent=4))

    # --------------------------------------------------
    # Save Model + Schema Metadata
    # --------------------------------------------------

    model_artifact = {
        "pipeline": pipeline,
        "threshold": best_threshold,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features
    }

    joblib.dump(model_artifact, f"{MODEL_DIR}/model.joblib")

    with open(f"{METRICS_DIR}/metrics.json", "w") as f:
        json.dump(results, f, indent=4)

    print("Training Complete.")


if __name__ == "__main__":
    main()