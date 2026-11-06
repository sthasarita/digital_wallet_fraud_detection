import sqlite3
from pathlib import Path
from datetime import datetime

#database location
BASE_DIR = Path(__file__).resolve().parent.parent

DATABASE_DIR = BASE_DIR / "database"
DATABASE_DIR.mkdir(exist_ok = True)

DATABASE_PATH = DATABASE_DIR / "walletguard.db"

#create database table
def init_db():
    with sqlite3.connect(DATABASE_PATH) as connection:
        cursor = connection.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            type TEXT NOT NULL,
            amount REAL NOT NULL,

            oldbalanceOrg REAL NOT NULL,
            newbalanceOrig REAL NOT NULL,

            oldbalanceDest REAL NOT NULL,
            newbalanceDest REAL NOT NULL,

            rf_probability REAL NOT NULL,
            rf_prediction INTEGER NOT NULL,

            ae_error REAL NOT NULL,
            ae_prediction INTEGER NOT NULL,

            risk_level TEXT NOT NULL,
            decision TEXT NOT NULL
            )
        """)

        connection.commit()


#save transaction

def save_transaction(transaction, result):

    rf_result = result["random_forest"]
    ae_result = result["autoencoder"]

    timestamp = datetime.now().isoformat(
        timespec="seconds"
    )

    values = (
        timestamp,
        transaction["type"],
        transaction["amount"],
        transaction["oldbalanceOrg"],
        transaction["newbalanceOrig"],
        transaction["oldbalanceDest"],
        transaction["newbalanceDest"],
        rf_result["fraud_probability"],
        rf_result["prediction"],
        ae_result["reconstruction_error"],
        ae_result["prediction"],
        result["risk_level"],
        result["decision"]
    )

    with sqlite3.connect(DATABASE_PATH) as connection:

        cursor = connection.cursor()

        cursor.execute("""
            INSERT INTO transactions (
                timestamp,
                type,
                amount,
                oldbalanceOrg,
                newbalanceOrig,
                oldbalanceDest,
                newbalanceDest,
                rf_probability,
                rf_prediction,
                ae_error,
                ae_prediction,
                risk_level,
                decision
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, values)

        connection.commit()

        return cursor.lastrowid



#retrieve transaction history
def get_transactions():
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.row_factory = sqlite3.Row

        cursor = connection.cursor()

        cursor.execute("""
            SELECT *
            FROM transactions
            ORDER BY id DESC
        """)


        rows = cursor.fetchall()

        return [dict(row) for row in rows]
