import joblib
import pandas as pd
import numpy as np

from app.services.feature_engineering import create_features
from app.config import settings


artifact = joblib.load(settings.MODEL_PATH)

pipeline = artifact["pipeline"]
threshold = artifact["threshold"]
numeric_features = artifact["numeric_features"]
categorical_features = artifact["categorical_features"]


def predict_transaction(transaction_dict):

    df = pd.DataFrame([transaction_dict])

    df = create_features(df)

    # 🔥 Ensure all required columns exist
    for col in numeric_features + categorical_features:
        if col not in df.columns:
            df[col] = np.nan

    # 🔥 Reorder columns exactly like training
    df = df[numeric_features + categorical_features]

    # 🔥 Force numeric types correctly
    for col in numeric_features:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    probability = pipeline.predict_proba(df)[0][1]

    if probability >= threshold:
        risk_level = "HIGH"
    elif probability >= threshold * 0.5:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return {
        "fraud_probability": float(probability),
        "risk_level": risk_level
    }