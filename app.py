import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BFSI Stock Predictor",
    page_icon="📈",
    layout="wide",
)

st.title("📈 BFSI Stock Movement Predictor")
st.markdown(
    "Upload your BFSI stock data, train the model, and predict next-day price direction."
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")
test_size   = st.sidebar.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
n_estimators = st.sidebar.slider("Random Forest Trees", 50, 500, 100, 50)
random_state = st.sidebar.number_input("Random State", value=42)

# ── Feature list (must match your notebook) ───────────────────────────────────
FEATURES = [
    "Open Price", "High Price", "Low Price", "Close Price",
    "WAP", "No.of Shares", "No. of Trades", "Deliverable Quantity",
    "Spread High-Low", "Spread Close-Open",
    "Daily Return", "Volatility", "Volume Per Trade",
    "Price_vs_MA5",
]

# ── Helper: feature engineering (mirrors notebook) ───────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Deliverable Quantity"].fillna(df["Deliverable Quantity"].median(), inplace=True)
    df.dropna(subset=["Date"], inplace=True)

    df["% Deli. Qty to Traded Qty"] = (
        df["Deliverable Quantity"] / df["No.of Shares"].replace(0, np.nan)
    ) * 100
    df["% Deli. Qty to Traded Qty"].fillna(0, inplace=True)

    df.sort_values(by=["Source.Name", "Date"], inplace=True)
    df["Next Close"] = df.groupby("Source.Name")["Close Price"].shift(-1)
    df["Target"] = (df["Next Close"] > df["Close Price"]).astype(int)

    df["Daily Return"]   = (df["Close Price"] - df["Open Price"]) / df["Open Price"]
    df["Volatility"]     = (df["High Price"] - df["Low Price"]) / df["Open Price"]
    df["Volume Per Trade"] = df["No.of Shares"] / df["No. of Trades"]

    df["MA5"]  = df.groupby("Source.Name")["Close Price"].transform(
        lambda x: x.rolling(5).mean()
    )
    df["MA10"] = df.groupby("Source.Name")["Close Price"].transform(
        lambda x: x.rolling(10).mean()
    )
    df["Price_vs_MA5"] = df["Close Price"] / df["MA5"]

    df.dropna(inplace=True)
    return df

# ── Section 1: Upload data ────────────────────────────────────────────────────
st.header("1️⃣  Upload Data")
uploaded = st.file_uploader("Upload your BFSI Excel file (.xlsx)", type=["xlsx"])

if uploaded:
    raw_df = pd.read_excel(uploaded)
    st.success(f"Loaded {len(raw_df):,} rows × {raw_df.shape[1]} columns")

    with st.expander("🔍 Preview raw data"):
        st.dataframe(raw_df.head(20))

    # ── Section 2: Feature engineering ───────────────────────────────────────
    st.header("2️⃣  Feature Engineering")
    with st.spinner("Engineering features…"):
        df = engineer_features(raw_df)

    st.success(f"After cleaning & feature engineering: {len(df):,} rows")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows",    f"{len(df):,}")
    col2.metric("Up Days (1)",   f"{df['Target'].sum():,}")
    col3.metric("Down Days (0)", f"{(df['Target']==0).sum():,}")

    with st.expander("📊 Engineered features sample"):
        st.dataframe(df[FEATURES + ["Target"]].head(20))

    # ── Section 3: EDA ────────────────────────────────────────────────────────
    st.header("3️⃣  Exploratory Data Analysis")

    tab1, tab2, tab3 = st.tabs(["Close Price History", "Correlation Heatmap", "Feature Distributions"])

    with tab1:
        stocks = df["Source.Name"].unique().tolist()
        chosen = st.selectbox("Select Stock", stocks)
        sub = df[df["Source.Name"] == chosen]
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(sub["Date"], sub["Close Price"], linewidth=1)
        ax.set_title(f"{chosen} — Close Price")
        ax.set_xlabel("Date"); ax.set_ylabel("Close Price (₹)")
        st.pyplot(fig)

    with tab2:
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        corr = df[FEATURES + ["Target"]].corr()
        sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax2)
        ax2.set_title("Feature Correlation Matrix")
        st.pyplot(fig2)

    with tab3:
        feat_sel = st.selectbox("Feature to plot", FEATURES)
        fig3, ax3 = plt.subplots(figsize=(8, 3))
        df[feat_sel].hist(bins=50, ax=ax3, color="steelblue", edgecolor="white")
        ax3.set_title(f"Distribution of {feat_sel}")
        st.pyplot(fig3)

    # ── Section 4: Train model ────────────────────────────────────────────────
    st.header("4️⃣  Train Model")

    if st.button("🚀 Train Random Forest"):
        X = df[FEATURES]
        y = df["Target"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=int(random_state),
            stratify=y,
        )

        with st.spinner("Training…"):
            model = RandomForestClassifier(
                n_estimators=int(n_estimators),
                random_state=int(random_state),
                class_weight="balanced",
            )
            model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # Save model to session state
        st.session_state["model"] = model
        st.session_state["features"] = FEATURES

        st.success(f"✅ Model trained! Accuracy: **{acc:.2%}**")

        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose().round(3))

        with col_b:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig4, ax4 = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax4)
            ax4.set_xlabel("Predicted"); ax4.set_ylabel("Actual")
            st.pyplot(fig4)

        # Feature importance
        st.subheader("🌟 Feature Importance")
        importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=True)
        fig5, ax5 = plt.subplots(figsize=(8, 5))
        importances.plot(kind="barh", ax=ax5, color="steelblue")
        ax5.set_title("Feature Importances")
        st.pyplot(fig5)

        # Save model button
        joblib.dump(model, "/tmp/bfsi_model.pkl")
        with open("/tmp/bfsi_model.pkl", "rb") as f:
            st.download_button(
                "⬇️ Download Trained Model (.pkl)",
                f,
                file_name="bfsi_model.pkl",
                mime="application/octet-stream",
            )

    # ── Section 5: Predict on new data ────────────────────────────────────────
    st.header("5️⃣  Predict — Manual Input")

    if "model" in st.session_state:
        st.markdown("Enter today's stock data to predict tomorrow's direction:")

        input_cols = st.columns(3)
        user_inputs = {}
        for idx, feat in enumerate(FEATURES):
            with input_cols[idx % 3]:
                user_inputs[feat] = st.number_input(feat, value=0.0, format="%.4f")

        if st.button("🔮 Predict"):
            input_df = pd.DataFrame([user_inputs])
            prediction = st.session_state["model"].predict(input_df)[0]
            proba = st.session_state["model"].predict_proba(input_df)[0]

            if prediction == 1:
                st.success(f"📈 **UP** — Model predicts the price will RISE tomorrow  (confidence: {proba[1]:.1%})")
            else:
                st.error(f"📉 **DOWN** — Model predicts the price will FALL tomorrow  (confidence: {proba[0]:.1%})")
    else:
        st.info("Train the model first (Section 4) to unlock predictions.")

else:
    st.info("👆 Upload your BFSI Excel file to get started.")
