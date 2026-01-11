Stock Price Prediction using Linear Regression

This project builds and evaluates a simple machine-learning model to predict stock closing prices using historical data and technical indicators.
The script pulls market data from Yahoo Finance, trains a linear regression model, evaluates prediction quality on a test set, and forecasts the next day’s price.

Features

The script computes several technical indicators used as model inputs:

10-day moving average (MA10)

30-day moving average (MA30)

Volatility (10-day rolling standard deviation of returns)

Trading Volume

The model is trained using:

LinearRegression from scikit-learn

Standardized features (StandardScaler)

Walk-forward chronological train/test split (no shuffling)

What the Program Does

Downloads price data from Yahoo Finance for a user-entered ticker.

Creates predictive features based on price and volume history.

Splits data into train and test segments.

Fits a regression model and predicts stock prices on unseen data.

Prints model accuracy using:

Mean Absolute Error (MAE)

Root Mean Squared Error (RMSE)

Trains on the full dataset and forecasts the next trading day’s closing price.

Plots actual vs predicted prices and marks the forecasted point.
