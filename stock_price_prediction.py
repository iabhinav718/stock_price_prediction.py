import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import yfinance as yf

def main():
    ticker = input("Enter stock ticker (E.g., AAPL, TSLA, RELIANCE.NS): ").strip()

    data = yf.download(ticker, start="2015-01-01", end="2024-01-01")
    if data.empty:
        print("Invalid ticker or no data available.")
        return

    df = data.copy()
    df["Return"] = df["Close"].pct_change()
    df["MA10"] = df["Close"].rolling(window=10).mean()
    df["MA30"] = df["Close"].rolling(window=30).mean()
    df["Volatility"] = df["Return"].rolling(10).std()
    df = df.dropna()

    features = ["MA10", "MA30", "Volatility", "Volume"]
    X = df[features]
    y = df["Close"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"MAE:  {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")

    plt.figure(figsize=(12,6))
    plt.plot(y_test.index, y_test.values, label="Actual Price")
    plt.plot(y_test.index, y_pred,        label="Predicted Price")
    plt.title(f"{ticker} Stock Price Prediction (Linear Regression)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()

    scaler_full = StandardScaler()
    X_scaled_full = scaler_full.fit_transform(X)

    model_full = LinearRegression()
    model_full.fit(X_scaled_full, y)

    last_scaled = scaler_full.transform([X.iloc[-1].values])
    tomorrow_price = float(model_full.predict(last_scaled)[0])
    last_actual = float(y.iloc[-1])

    print("------------------------------------------")
    print(f"Last Actual Close Price:    {last_actual:.2f}")
    print(f"Predicted Tomorrow Price: {tomorrow_price:.2f}")
    print("------------------------------------------")

    next_date = y.index[-1] + pd.Timedelta(days=1)
    plt.scatter(next_date, tomorrow_price, color='red', label="Tomorrow (Forecast)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
