import numpy as np
import pandas as pd
#Feature configuration
NUMERIC_FEATURES = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "balance_change_orig",
    "balance_change_dest"
]

EXPECTED_TRANSACTION_TYPES = {
    "CASH_IN",
    "CASH_OUT",
    "DEBIT",
    "PAYMENT",
    "TRANSFER"
}

#validate transaction input
def validate_transactions(transaction):
    required_fields = [
        "type",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest"
    ]

    #check whether all required fields exist
    missing_fields = [
        field for field in required_fields
        if field not in transaction
    ]

    if missing_fields:
        raise ValueError(
            f"Missing transaction fields:{missing_fields}"
        )

    #validate transaction type
    transaction_type = transaction["type"]

    if transaction_type not in EXPECTED_TRANSACTION_TYPES:
        raise ValueError(
            f"Unsupported transaction type:{transaction_type}"
        )

    #validate numerical values
    numerical_values = {}
    for field in required_fields[1:]:
        try:
            value = float(transaction[field])
        except (TypeError, ValueError):
            raise ValueError(
                f"{field} must be a valid number."
            )

        if not np.isfinite(value):
            raise ValueError(
                f"{field} must be finite."
            )

        if value < 0:
            raise ValueError(
                f"{field} cann't be negative."
            )
        numerical_values[field] = value
    return transaction_type, numerical_values

#feature engineering
def engineer_feature(transaction):
    transaction_type, values = validate_transactions(
        transaction
    )

    #calculate balance changes
    balance_change_orig = (
        values["oldbalanceOrg"] - values["newbalanceOrig"]

    )

    balance_change_dest = (
        values["newbalanceDest"] - values["oldbalanceDest"]
    )

    #collect numerical features in training order
    numerical_features = np.array([
        values["amount"],
        values["oldbalanceOrg"],
        values["newbalanceOrig"],
        values["oldbalanceDest"],
        values["newbalanceDest"],
        balance_change_orig,
        balance_change_dest
    ], dtype = np.float64)

    return transaction_type, numerical_features

#Apply training - time transformations
def transform_numerical_features(numerical_features):
    transformed = numerical_features.copy()

    #log transformation for the first five features
    transformed[:5] = np.log1p(transformed[:5])

    #Signed log transformation for balance changes
    transformed[5:] = (
        np.sign(transformed[5:]) *np.log1p(np.abs(transformed[5:]))
    )

    return transformed.astype(np.float32)

#convert transactions into model inputs

def preprocess_transaction(
        transaction,
        encoder,
        scaler
):
    #feature engineering
    transaction_type , numerical_features = (
        engineer_feature(transaction)
    )

    #apply logarithmic transformations
    transformed_numerical = (
        transform_numerical_features(
            numerical_features
        )
    )

    #One-hot encode transaction type
    encoded_type = encoder.transform(
    pd.DataFrame({"type": [transaction_type]})
    )
    encoded_type = np.asarray(
        encoded_type,
        dtype = np.float32
    )

    #Combine encoded type and numerical features
    combined_features = np.concatenate(
        [
            encoded_type,
            transformed_numerical.reshape(1,-1)
        ],
        axis = 1
    )

    #confirm the expected 12-feature input
    if combined_features.shape[1] != 12:
        raise ValueError(
            f"Expected 12 features, got "
            f"{combined_features.shape[1]}"
        )

    #Random forest receives unscaled features
    rf_input = combined_features.copy()

    #AE3 receives scaled numerical features
    ae_input = combined_features.copy()

    ae_input[:, 5:] = scaler.transform(
        combined_features[:,5:]
    )
    return rf_input, ae_input