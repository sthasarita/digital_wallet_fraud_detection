
import streamlit as st
import requests
import pandas as pd


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

BASE_API_URL = "http://127.0.0.1:8000"

PREDICT_URL = f"{BASE_API_URL}/predict"
HISTORY_URL = f"{BASE_API_URL}/transactions"

st.set_page_config(
    page_title="WalletGuard | Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# 2. SESSION STATE
# --------------------------------------------------

# Keep only the latest prediction for display.
# Transaction history now comes from SQLite.

if "last_result" not in st.session_state:
    st.session_state.last_result = None


# --------------------------------------------------
# 3. FETCH TRANSACTIONS FROM FASTAPI
# --------------------------------------------------

def fetch_transactions():

    try:

        response = requests.get(
            HISTORY_URL,
            timeout=15
        )

        response.raise_for_status()

        data = response.json()

        return data["transactions"], None

    except requests.exceptions.ConnectionError:

        return None, "Cannot connect to FastAPI. Please start the backend."

    except requests.exceptions.Timeout:

        return None, "The database request timed out."

    except requests.exceptions.RequestException as error:

        return None, f"API request failed: {error}"

    except (KeyError, ValueError):

        return None, "Received an invalid response from the API."


# --------------------------------------------------
# 4. CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

[data-testid="stSidebar"] {
    background-color: #111827;
}

[data-testid="stSidebar"] * {
    color: white;
}

.dashboard-title {
    font-size: 30px;
    font-weight: 700;
    color: #172554;
}

.dashboard-subtitle {
    color: #64748b;
    font-size: 15px;
}

.metric-card {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
    min-height: 115px;
}

.metric-label {
    color: #64748b;
    font-size: 14px;
}

.metric-value {
    color: #172554;
    font-size: 27px;
    font-weight: 700;
    margin-top: 8px;
}

.section-title {
    color: #172554;
    font-size: 21px;
    font-weight: 650;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 5. SIDEBAR NAVIGATION
# --------------------------------------------------

with st.sidebar:

    st.markdown("# 🛡️ WalletGuard")
    st.caption("Digital Wallet Fraud Detection")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Dashboard",
            "Analyze Transaction",
            "Transaction History"
        ],
        label_visibility="visible"
    )

    st.divider()

    st.caption("MACHINE LEARNING MODELS")
    st.write("🌲 Random Forest")
    st.write("🔍 Autoencoder")

    st.divider()

    st.caption("Fraud Monitoring System")
    st.caption("Prototype Version 1.0")


# --------------------------------------------------
# 6. HELPER FUNCTION: METRIC CARD
# --------------------------------------------------

def metric_card(label, value):

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# --------------------------------------------------
# 7. FETCH DATABASE RECORDS
# --------------------------------------------------

transactions, fetch_error = fetch_transactions()


# --------------------------------------------------
# 8. DASHBOARD PAGE
# --------------------------------------------------

if page == "Dashboard":

    st.markdown(
        '<div class="dashboard-title">Fraud Monitoring Dashboard</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="dashboard-subtitle">'
        'Monitor analyzed transactions and machine learning risk assessments.'
        '</div>',
        unsafe_allow_html=True
    )

    st.write("")

    if fetch_error:

        st.error(fetch_error)

    else:

        total = len(transactions)

        flagged = sum(
            1 for item in transactions
            if item["decision"] == "Flag for review"
        )

        normal = total - flagged

        flagged_rate = (
            (flagged / total) * 100
            if total > 0 else 0
        )

        # Overview metrics

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            metric_card("Analyzed Transactions", total)

        with col2:
            metric_card("Flagged Transactions", flagged)

        with col3:
            metric_card("Unflagged Transactions", normal)

        with col4:
            metric_card("Flagged Rate", f"{flagged_rate:.2f}%")

        st.write("")
        st.divider()

        # Charts

        st.markdown(
            '<div class="section-title">Transaction Analytics</div>',
            unsafe_allow_html=True
        )

        if total > 0:

            history_df = pd.DataFrame(transactions)

            col1, col2 = st.columns(2)

            with col1:

                st.subheader("Risk Distribution")

                risk_counts = pd.DataFrame({
                    "Category": ["Flagged", "Unflagged"],
                    "Transactions": [flagged, normal]
                })

                st.bar_chart(
                    risk_counts.set_index("Category")
                )

            with col2:

                st.subheader("Transaction Types")

                type_counts = (
                    history_df["type"]
                    .value_counts()
                    .rename_axis("Transaction Type")
                    .to_frame("Count")
                )

                st.bar_chart(type_counts)

            st.divider()

            st.subheader("Recent Transactions")

            recent_df = history_df.head(5).copy()

            recent_df["rf_probability"] = (
                recent_df["rf_probability"] * 100
            ).round(2).astype(str) + "%"

            st.dataframe(
                recent_df[
                    [
                        "timestamp",
                        "type",
                        "amount",
                        "rf_probability",
                        "risk_level",
                        "decision"
                    ]
                ],
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "No transactions have been analyzed yet. "
                "Go to Analyze Transaction to get started."
            )


# --------------------------------------------------
# 9. ANALYZE TRANSACTION PAGE
# --------------------------------------------------

elif page == "Analyze Transaction":

    st.markdown(
        '<div class="dashboard-title">Transaction Analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="dashboard-subtitle">'
        'Enter transaction details to evaluate its fraud risk.'
        '</div>',
        unsafe_allow_html=True
    )

    st.write("")

    with st.form("transaction_form"):

        st.subheader("Transaction Information")

        col1, col2 = st.columns(2)

        with col1:

            transaction_type = st.selectbox(
                "Transaction Type",
                [
                    "CASH_IN",
                    "CASH_OUT",
                    "DEBIT",
                    "PAYMENT",
                    "TRANSFER"
                ]
            )

            amount = st.number_input(
                "Transaction Amount",
                min_value=0.0,
                value=5000.0,
                step=100.0
            )

            oldbalance_org = st.number_input(
                "Sender's Previous Balance",
                min_value=0.0,
                value=10000.0,
                step=100.0
            )

        with col2:

            newbalance_orig = st.number_input(
                "Sender's New Balance",
                min_value=0.0,
                value=5000.0,
                step=100.0
            )

            oldbalance_dest = st.number_input(
                "Receiver's Previous Balance",
                min_value=0.0,
                value=2000.0,
                step=100.0
            )

            newbalance_dest = st.number_input(
                "Receiver's New Balance",
                min_value=0.0,
                value=7000.0,
                step=100.0
            )

        submitted = st.form_submit_button(
            "🔍 Analyze Transaction",
            use_container_width=True
        )

    # Send transaction to FastAPI

    if submitted:

        transaction_data = {
            "type": transaction_type,
            "amount": amount,
            "oldbalanceOrg": oldbalance_org,
            "newbalanceOrig": newbalance_orig,
            "oldbalanceDest": oldbalance_dest,
            "newbalanceDest": newbalance_dest
        }

        try:

            with st.spinner("Analyzing transaction..."):

                response = requests.post(
                    PREDICT_URL,
                    json=transaction_data,
                    timeout=60
                )

            if response.status_code == 200:

                result = response.json()

                # Store only the latest prediction in session state.
                # FastAPI has already saved the transaction in SQLite.

                st.session_state.last_result = result

                st.success(
                    f"Transaction analysis completed! "
                    f"Transaction ID: {result['transaction_id']}"
                )

                # Refresh the page so database-backed records
                # are available immediately.

                st.rerun()

            else:

                st.error(
                    f"API Error: {response.status_code}"
                )

                st.write(response.text)

        except requests.exceptions.ConnectionError:

            st.error(
                "Cannot connect to FastAPI. "
                "Please make sure the backend server is running."
            )

        except requests.exceptions.Timeout:

            st.error(
                "The prediction request took too long."
            )

        except requests.exceptions.RequestException as error:

            st.error(f"Request failed: {error}")

    # Display latest prediction

    if st.session_state.last_result is not None:

        result = st.session_state.last_result

        rf_result = result["random_forest"]
        ae_result = result["autoencoder"]

        st.divider()

        st.subheader("Prediction Results")

        st.write(
            f"Transaction ID: **{result['transaction_id']}**"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Random Forest Fraud Probability",
                f"{rf_result['fraud_probability'] * 100:.2f}%"
            )

            st.write(
                "Prediction:",
                "🔴 Fraud"
                if rf_result["prediction"] == 1
                else "🟢 Normal"
            )

        with col2:

            st.metric(
                "Autoencoder Reconstruction Error",
                f"{ae_result['reconstruction_error']:.6f}"
            )

            st.write(
                "Prediction:",
                "⚠️ Anomaly Detected"
                if ae_result["prediction"] == 1
                else "🟢 Normal"
            )

        st.divider()

        st.subheader("Final Risk Assessment")

        if result["decision"] == "Flag for review":

            st.error(
                f"⚠️ {result['risk_level']} — "
                f"{result['decision']}"
            )

        else:

            st.success(
                f"✅ {result['risk_level']} — "
                f"{result['decision']}"
            )


# --------------------------------------------------
# 10. TRANSACTION HISTORY PAGE
# --------------------------------------------------

elif page == "Transaction History":

    st.markdown(
        '<div class="dashboard-title">Transaction History</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="dashboard-subtitle">'
        'Review all transactions saved in the database.'
        '</div>',
        unsafe_allow_html=True
    )

    st.write("")

    if fetch_error:

        st.error(fetch_error)

    elif transactions:

        history_df = pd.DataFrame(transactions)

        # Convert RF probability into percentage for display

        history_df["rf_probability"] = (
            history_df["rf_probability"] * 100
        ).round(2)

        st.metric(
            "Total Saved Transactions",
            len(history_df)
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        # Download database history as CSV

        csv_data = history_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇️ Download Transaction History",
            data=csv_data,
            file_name="walletguard_transaction_history.csv",
            mime="text/csv"
        )

    else:

        st.info(
            "No transaction history available yet."
        )
