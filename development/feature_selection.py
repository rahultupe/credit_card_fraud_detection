
def feature_selection(df):

    column_to_drop=([
        "transaction_id",
        "amount_usd",
        "merchant_category",
        "hours_since_last_txn",
        "distance_from_home_km",
        "card_age_months",
        "account_balance_usd",
        "is_fraud"
    ])

    X = df.drop(column_to_drop,axis=1,errors="ignore")
    if "is_fraud" in df.columns:
        y=df["is_fraud"]

        return X,y
    return X

    