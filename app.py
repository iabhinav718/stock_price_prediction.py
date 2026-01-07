import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import matplotlib.pyplot as plt

st.title("Stock Price Prediction (Tomorrow Forecast) 📈")

# Text input for stock ticker
ticker = st.text_input("Enter stock ticker (e.g., AAPL, TSLA, RELIANCE.NS)", value="AAPL")

if st.button("Predict Tomorrow"):
    data = yf.download(ticker, period="5y", interval="1d")

    if data.empty:
        st.error("Invalid ticker or no data available.")
    else:
        df = data.copy()
        df["Return"] = df["Close"].pct_change()
        df["MA10"] = df["Close"].rolling(window=10).mean()
        df["MA30"] = df["Close"].rolling(window=30).mean()
        df["Volatility"] = df["Return"].rolling(10).std()
        df = df.dropna()

        features = ["MA10", "MA30", "Volatility", "Volume"]
        X = df[features]
        y = df["Close"]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = LinearRegression()
        model.fit(X_scaled, y)

        last_scaled = scaler.transform([X.iloc[-1].values])
        tomorrow_price = float(model.predict(last_scaled)[0])
        last_actual = float(y.iloc[-1])

        st.success(f"Last Actual Close Price: {last_actual:.2f}")
        st.success(f"Predicted Tomorrow Price: {tomorrow_price:.2f}")

        # Plot
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df.index[-200:], df["Close"].tail(200), label="Actual Price")
        ax.scatter(df.index[-1] + pd.Timedelta(days=1), tomorrow_price, color="red", label="Tomorrow (Predicted)")
        ax.legend()
        st.pyplot(fig)
