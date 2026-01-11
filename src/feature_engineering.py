import pandas as pd
from datetime import datetime

def create_features(df, mode="train"):
    """
    mode = 'train'  → training pipeline
    mode = 'infer'  → API inference
    """

    df = df.copy()

    # ---------- TENURE LOGIC ----------
    if mode == "train":
        if "date_of_registration" not in df.columns:
            raise ValueError("Training data must include 'date_of_registration'")

        df["date_of_registration"] = pd.to_datetime(df["date_of_registration"])
        today = datetime.today()
        df["tenure_days"] = (today - df["date_of_registration"]).dt.days
        df["tenure_months"] = df["tenure_days"] // 30

    elif mode == "infer":
        if "tenure_months" not in df.columns:
            raise ValueError("Inference data must include 'tenure_months'")

        df["tenure_days"] = df["tenure_months"] * 30

    else:
        raise ValueError("mode must be 'train' or 'infer'")

    # ---------- USAGE NORMALIZATION ----------
    df["calls_per_month"] = df["calls_made"] / (df["tenure_months"] + 1)
    df["sms_per_month"] = df["sms_sent"] / (df["tenure_months"] + 1)
    df["data_per_month"] = df["data_used"] / (df["tenure_months"] + 1)

    # ---------- VALUE FLAG ----------
    if "estimated_salary" in df.columns:
        df["high_value_customer"] = (
            df["estimated_salary"] > df["estimated_salary"].median()
        ).astype(int)

    # Drop raw date
    df.drop(columns=["date_of_registration"], errors="ignore", inplace=True)

    return df
