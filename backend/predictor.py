
import json
import joblib
import numpy as np
import tensorflow as tf

from pathlib import Path

from backend.preprocessing import preprocess_transaction


# --------------------------------------------------
# 1. Locate model artifacts
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"


# --------------------------------------------------
# 2. Load trained models and preprocessing objects
# --------------------------------------------------

random_forest = joblib.load(
    MODEL_DIR / "random_forest.joblib"
)

autoencoder = tf.keras.models.load_model(
    MODEL_DIR / "autoencoder_ae3.keras"
)

encoder = joblib.load(
    MODEL_DIR / "onehot_encoder.joblib"
)

scaler = joblib.load(
    MODEL_DIR / "standard_scaler.joblib"
)


# --------------------------------------------------
# 3. Load model metadata
# --------------------------------------------------

with open(MODEL_DIR / "model_metadata.json", "r") as file:
    metadata = json.load(file)


RF_THRESHOLD = metadata["models"]["random_forest"]["threshold"]

AE_THRESHOLD = metadata["models"]["autoencoder_ae3"]["threshold"]


# --------------------------------------------------
# 4. Prediction function
# --------------------------------------------------

def predict_transaction(transaction):

    # Preprocess incoming transaction
    rf_input, ae_input = preprocess_transaction(
        transaction,
        encoder,
        scaler
    )

#for certain time 1
    print("\nRandom Forest input:")
    print(rf_input)

    print("\nAutoencoder input:")
    print(ae_input)

    print("\nInput shapes:")
    print("RF:", rf_input.shape)
    print("AE:", ae_input.shape)

    #up to here 1

    # Random Forest fraud probability
    rf_probability = float(
        random_forest.predict_proba(rf_input)[0, 1]
    )

    # Random Forest classification
    rf_prediction = int(
        rf_probability >= RF_THRESHOLD
    )

    # Autoencoder reconstruction
    reconstructed = autoencoder.predict(
        ae_input,
        verbose=0
    )

    # Calculate mean squared reconstruction error
    reconstruction_error = float(
        np.mean(
            np.square(ae_input - reconstructed)
        )
    )

    # Autoencoder anomaly classification
    ae_prediction = int(
        reconstruction_error >= AE_THRESHOLD
    )

    # Final result: Random Forest is the primary model
    if rf_prediction == 1:
        risk_level = "High Risk"
        decision = "Flag for review"

    elif ae_prediction == 1:
        risk_level = "Anomaly Detected"
        decision = "Flag for review"

    else:
        risk_level = "Low Risk"
        decision = "No fraud flag"

    return {
        "random_forest": {
            "fraud_probability": round(rf_probability, 6),
            "prediction": rf_prediction
        },

        "autoencoder": {
            "reconstruction_error": round(
                reconstruction_error, 6
            ),
            "prediction": ae_prediction
        },

        "risk_level": risk_level,
        "decision": decision
    }
