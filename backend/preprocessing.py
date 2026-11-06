
import numpy as np
import pandas as pd


def preprocess_transaction(transaction, encoder, scaler):
    """
    Preprocess one transaction for the Random Forest and Autoencoder AE3.

    Random Forest:
        5 one-hot encoded features + 7 raw numerical features

    Autoencoder AE3:
        5 one-hot encoded features + 7 standardized numerical features

    Returns:
        rf_input: Input for Random Forest, shape (1, 12)
        ae_input: Input for Autoencoder AE3, shape (1, 12)
    """

    # 1. Define the required transaction fields
    required_fields = [
        "type",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest"
    ]

    # 2. Check whether all required fields are provided
    for field in required_fields:
        if field not in transaction:
            raise ValueError(f"Missing required field: {field}")

    # 3. Extract transaction type
    transaction_type = transaction["type"]

    if not isinstance(transaction_type, str) or not transaction_type.strip():
        raise ValueError("Transaction type must be a non-empty string.")

    # 4. Extract and validate numerical features
    numerical_fields = [
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest"
    ]

    numerical_values = []

    for field in numerical_fields:
        try:
            value = float(transaction[field])
        except (TypeError, ValueError):
            raise ValueError(f"{field} must be a valid number.")

        if not np.isfinite(value):
            raise ValueError(f"{field} must be finite.")

        if value < 0:
            raise ValueError(f"{field} cannot be negative.")

        numerical_values.append(value)

    (
        amount,
        oldbalanceOrg,
        newbalanceOrig,
        oldbalanceDest,
        newbalanceDest
    ) = numerical_values

    # 5. Check whether transaction type was present during training
    known_types = encoder.categories_[0]

    if transaction_type not in known_types:
        raise ValueError(
            f"Unsupported transaction type: {transaction_type}"
        )

    # 6. One-hot encode transaction type
    type_df = pd.DataFrame({
        "type": [transaction_type]
    })

    encoded_type = encoder.transform(type_df)

    # Convert sparse output to a dense NumPy array if necessary
    if hasattr(encoded_type, "toarray"):
        encoded_type = encoded_type.toarray()

    # 7. Calculate the two engineered numerical features
    balance_change_orig = oldbalanceOrg - newbalanceOrig
    balance_change_dest = newbalanceDest - oldbalanceDest

    # 8. Arrange numerical features in the exact training order
    numerical_features = np.array([[
        amount,
        oldbalanceOrg,
        newbalanceOrig,
        oldbalanceDest,
        newbalanceDest,
        balance_change_orig,
        balance_change_dest
    ]], dtype=np.float32)

    # 9. Combine 5 one-hot columns and 7 numerical columns
    combined_features = np.hstack([
        encoded_type,
        numerical_features
    ]).astype(np.float32)

    # 10. Random Forest receives the raw numerical values
    rf_input = combined_features.copy()

    # 11. Autoencoder receives standardized numerical values
    ae_input = combined_features.copy()

    # Scale only columns 5 through 11
    # Do not scale the first 5 one-hot encoded columns
    # Do not apply log1p()
    ae_input[:, 5:] = scaler.transform(
        ae_input[:, 5:]
    )

    return rf_input, ae_input
