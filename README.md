Stock Price Prediction (Linear Regression)

This project predicts daily stock closing prices using historical data from Yahoo Finance.
A linear regression model is trained using basic technical indicators.

Features Used      
 
10-Day Moving Average (MA10)

30-Day Moving Average (MA30)

Volatility (10-day rolling standard deviation)

Trading Volume

What the Script Does

Downloads price history for a user-entered ticker.

Builds features and splits data chronologically.

Trains a linear regression model with standardized inputs.

Evaluates predictions with MAE and RMSE.

Trains on all data and forecasts the next day’s closing price.

Plots actual vs predicted prices and shows the forecast point
