import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # -----------------------------
    # Clean Amount
    # -----------------------------
    if "Amount" in df.columns:
        df["Amount"] = (
            df["Amount"]
            .astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
        )
        df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")

    # -----------------------------
    # Extract Hour from Time
    # -----------------------------
    if "Time" in df.columns:
        df["Hour"] = (
            df["Time"]
            .astype(str)
            .str.split(":")
            .str[0]
        )
        df["Hour"] = pd.to_numeric(df["Hour"], errors="coerce")

    # -----------------------------
    # Night Transaction Flag
    # -----------------------------
    if "Hour" in df.columns:
        df["is_night_tx"] = df["Hour"].apply(
            lambda x: 1 if pd.notnull(x) and (x < 6 or x > 22) else 0
        )

    # -----------------------------
    # Drop columns we don’t want raw
    # -----------------------------
    columns_to_drop = [
        "Time"
    ]

    df = df.drop(columns=[c for c in columns_to_drop if c in df.columns])

    # -----------------------------
    # Final safety: force numeric where possible
    # -----------------------------
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            pass

    return df