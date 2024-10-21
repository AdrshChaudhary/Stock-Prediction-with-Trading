#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import vectorbt as vbt
import matplotlib.pyplot as plt
from alpaca_trade_api.rest import REST

# Streamlit App Layout
st.title('Stock Prediction and Trading App')

# User Inputs
symbol = st.text_input("Enter Stock Symbol (e.g., AAPL):", "AAPL")
start_date = st.date_input("Start Date:", pd.to_datetime('2020-01-01'))
end_date = st.date_input("End Date:", pd.to_datetime('2024-01-01'))
trade_quantity = st.number_input("Trade Quantity:", value=1, min_value=1)

# Download stock data
@st.cache_data
def load_data(symbol, start_date, end_date):
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    return stock_data

stock_data = load_data(symbol, start_date, end_date)

# Calculate moving averages
stock_data['MA50'] = stock_data['Close'].rolling(window=50).mean()
stock_data['MA200'] = stock_data['Close'].rolling(window=200).mean()

# Drop any NaN values
stock_data.dropna(inplace=True)

# Shift the target variable (Close price) to predict the next day
stock_data['Prediction'] = stock_data['Close'].shift(-1)
stock_data.dropna(inplace=True)

# Visualization: Historical Prices with Moving Averages
st.subheader('Historical Prices with Moving Averages')
plt.figure(figsize=(14, 7))
plt.plot(stock_data['Close'], label='Close Price', color='blue')
plt.plot(stock_data['MA50'], label='50-Day MA', color='orange')
plt.plot(stock_data['MA200'], label='200-Day MA', color='red')
plt.title(f'{symbol} Stock Price with Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
st.pyplot(plt)

# Define features and target
X = stock_data[['Close', 'MA50', 'MA200']]
y = stock_data['Prediction']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the random forest model
rf = RandomForestRegressor()

# Define the hyperparameters for grid search
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model after grid search
best_rf = grid_search.best_estimator_

# Make predictions
predictions = best_rf.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
st.write(f'Best Model Parameters: {grid_search.best_params_}')
st.write(f'Mean Squared Error: {mse:.2f}')

# Visualization: Actual vs. Predicted Prices (Scatter Plot)
st.subheader('Actual vs Predicted Prices')
plt.figure(figsize=(14, 7))
plt.scatter(y_test.index, y_test.values, label='Actual Prices', color='blue', alpha=0.6, edgecolor='k')
plt.scatter(y_test.index, predictions, label='Predicted Prices', color='red', alpha=0.6, edgecolor='k')
plt.title(f'Actual vs Predicted {symbol} Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
st.pyplot(plt)

# Create a signal to buy if the model predicts an increase
signals = best_rf.predict(X) > X['Close'].values

# Backtest with vectorbt
try:
    portfolio = vbt.Portfolio.from_signals(
        close=stock_data['Close'].values,
        entries=signals,
        exits=~signals,
        freq='1D'
    )

    # Display backtest results
    st.subheader('Backtest Results')
    stats = portfolio.stats()
    st.write("Total Return: {:.2f}%".format(stats['total_return'] * 100))
    st.write("Sharpe Ratio: {:.2f}".format(stats['sharpe']))
    st.write("Maximum Drawdown: {:.2f}%".format(stats['max_drawdown'] * 100))
    st.write("Total Trades: {}".format(stats['total_trades']))

    # Portfolio Plot using vectorbt
    st.subheader('Portfolio Performance')
    fig = portfolio.total_return.plot()
    st.plotly_chart(fig)
except Exception as e:
    st.error(f"Error running backtest: {str(e)}")

# Alpaca API credentials
ALPACA_API_KEY = st.secrets["global"]["ALPACA_API_KEY"]
ALPACA_SECRET_KEY = st.secrets["global"]["ALPACA_SECRET_KEY"]
ALPACA_BASE_URL = 'https://paper-api.alpaca.markets'

# Initialize Alpaca REST API
alpaca = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, api_version='v2')

# Define the trading function
def trade(symbol, prediction, current_price):
    try:
        if prediction > current_price:
            # Buy signal
            alpaca.submit_order(
                symbol=symbol,
                qty=trade_quantity,
                side='buy',
                type='market',
                time_in_force='day'
            )
            return f"Buying {symbol} at {current_price:.2f}"
        else:
            # Sell signal
            alpaca.submit_order(
                symbol=symbol,
                qty=trade_quantity,
                side='sell',
                type='market',
                time_in_force='day'
            )
            return f"Selling {symbol} at {current_price:.2f}"
    except Exception as e:
        return f"Error executing trade: {e}"

# Example usage with the last predicted value
if st.button('Execute Trade'):
    last_prediction = best_rf.predict([X.iloc[-1]])[0]
    last_price = X.iloc[-1]['Close']
    result = trade(symbol, last_prediction, last_price)
    st.write(result)
