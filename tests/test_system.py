import os
import joblib
import pandas as pd


def test_model_file_exists():
    assert os.path.exists("models/model.joblib")


def test_model_loads():
    model = joblib.load("models/model.joblib")
    assert model is not None


def test_data_file_exists():
    assert os.path.exists("data/User0_credit_card_transactions.csv")


def test_data_loads():
    df = pd.read_csv("data/User0_credit_card_transactions.csv")
    assert len(df) > 0