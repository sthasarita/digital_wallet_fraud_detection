
from backend.preprocessing import preprocess_transaction
from backend.predictor import (
    predict_transaction,
    encoder,
    scaler
)

import numpy as np


sample_transaction = {
    "type": "TRANSFER",
    "amount": 5000,
    "oldbalanceOrg": 10000,
    "newbalanceOrig": 5000,
    "oldbalanceDest": 2000,
    "newbalanceDest": 7000
}

print("Starting transaction prediction...", flush=True)

# Generate both model inputs
rf_input, ae_input = preprocess_transaction(
    sample_transaction,
    encoder,
    scaler
)

print("\nRandom Forest input:")
print(rf_input)

print("\nAutoencoder input:")
print(ae_input)

print("\nInput shapes:")
print("RF:", rf_input.shape)
print("AE:", ae_input.shape)

# Verify dimensions
assert rf_input.shape == (1, 12)
assert ae_input.shape == (1, 12)

# Verify that RF receives the expected raw numerical features
expected_numerical = np.array(
    [5000, 10000, 5000, 2000, 7000, 5000, 5000],
    dtype=np.float32
)

np.testing.assert_allclose(
    rf_input[0, 5:],
    expected_numerical
)

# Verify AE numerical features are standardized
expected_scaled = scaler.transform(
    expected_numerical.reshape(1, -1)
)

np.testing.assert_allclose(
    ae_input[:, 5:],
    expected_scaled,
    rtol=1e-5,
    atol=1e-5
)

print("\nPreprocessing verification passed!", flush=True)

# Run both models
result = predict_transaction(sample_transaction)

print("\nPrediction completed!", flush=True)
print("Transaction prediction:")
print(result)
