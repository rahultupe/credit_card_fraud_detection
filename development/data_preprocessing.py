import types

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (OneHotEncoder,
                                   LabelEncoder,
                                   OrdinalEncoder,
                                   TargetEncoder
                                   )
from sklearn.preprocessing import (StandardScaler,
                                   MinMaxScaler)

def create_preprocessing_pipeline():

    # Numerical columns remaining after feature selection we are using standard scaler
    numerical_columns = [
        "txn_count_last_24h",
        "customer_age",
        "cvv_retry_count",
        "velocity_score",
        "time_of_day_hour",
    ]

    #minmax_scaler

    minmax_columns = ["day_of_week",
            "merchant_risk_score",
            "prior_disputes"
            ]

    # Categorical columns remaining after feature selection we are using one hot encoding
    categorical_columns = [
        "card_type",
        "auth_method",
        "channel",
        "device_type"
         ]

    #ordinal encoder

    ordinal_columns = [
                "ip_country_mismatch",
                "billing_shipping_mismatch",
                "is_ai_generated_scam_attempt",
                "is_foreign_transaction",
                "is_new_merchant",
                "used_vpn"
            ]

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    minmax_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler",MinMaxScaler())
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    ordinal_pipeline = Pipeline([
        ("imputer",SimpleImputer(strategy="mean")),
        ("encoder", OrdinalEncoder())
    ])

    preprocessing_pipeline = ColumnTransformer([
        ("numerical", numerical_pipeline, numerical_columns),
        ("minmax", minmax_pipeline, minmax_columns),
        ("categorical", categorical_pipeline, categorical_columns),
        ("ordinal", ordinal_pipeline, ordinal_columns)
    ])

    return preprocessing_pipeline