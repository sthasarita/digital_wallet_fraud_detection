
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal

from backend.predictor import predict_transaction
from backend.database import (
    init_db,
    save_transaction,
    get_transactions
)


# --------------------------------------------------
# 1. DATABASE INITIALIZATION
# --------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):

    # Create database table when backend starts
    init_db()

    yield


# --------------------------------------------------
# 2. CREATE FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="Digital Wallet Fraud Detection API",
    description="API for detecting suspicious digital wallet transactions.",
    version="1.0.0",
    lifespan=lifespan
)


# --------------------------------------------------
# 3. TRANSACTION DATA VALIDATION
# --------------------------------------------------

class Transaction(BaseModel):

    type: Literal[
        "CASH_IN",
        "CASH_OUT",
        "DEBIT",
        "PAYMENT",
        "TRANSFER"
    ]

    amount: float = Field(ge=0)
    oldbalanceOrg: float = Field(ge=0)
    newbalanceOrig: float = Field(ge=0)
    oldbalanceDest: float = Field(ge=0)
    newbalanceDest: float = Field(ge=0)


# --------------------------------------------------
# 4. HOME ENDPOINT
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "Digital Wallet Fraud Detection API is running!"
    }


# --------------------------------------------------
# 5. PREDICTION ENDPOINT
# --------------------------------------------------

@app.post("/predict")
def predict(transaction: Transaction):

    # Convert incoming transaction into dictionary
    transaction_data = transaction.model_dump()

    # Get predictions from trained ML models
    result = predict_transaction(transaction_data)

    # Save transaction and predictions in SQLite
    transaction_id = save_transaction(
        transaction_data,
        result
    )

    # Include database record ID in API response
    result["transaction_id"] = transaction_id

    return result


# --------------------------------------------------
# 6. TRANSACTION HISTORY ENDPOINT
# --------------------------------------------------

@app.get("/transactions")
def transaction_history():

    # Retrieve saved transactions from SQLite
    transactions = get_transactions()

    return {
        "total_transactions": len(transactions),
        "transactions": transactions
    }
