import pandas as pd

def clean_data(df):
    df = df.copy()

    # Safe drop
    df.drop(columns=["customer_id"], errors="ignore", inplace=True)

    # Missing values
    num_cols = df.select_dtypes(include="number").columns
    cat_cols = df.select_dtypes(exclude="number").columns

    df[num_cols] = df[num_cols].fillna(0)
    if len(cat_cols) > 0:
        df[cat_cols] = df[cat_cols].fillna("UNKNOWN")

    return df
