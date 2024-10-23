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
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(stock_data.index, stock_data['Close'], label='Close Price', color='blue')
ax.plot(stock_data.index, stock_data['MA50'], label='50-Day MA', color='orange')
ax.plot(stock_data.index, stock_data['MA200'], label='200-Day MA', color='red')
ax.set_title(f'{symbol} Stock Price with Moving Averages')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.legend()
st.pyplot(fig)
plt.close()

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

# Visualization: Actual vs. Predicted Prices
st.subheader('Actual vs Predicted Prices')
fig, ax = plt.subplots(figsize=(14, 7))
ax.scatter(y_test.index, y_test.values, label='Actual Prices', color='blue', alpha=0.6, edgecolor='k')
ax.scatter(y_test.index, predictions, label='Predicted Prices', color='red', alpha=0.6, edgecolor='k')
ax.set_title(f'Actual vs Predicted {symbol} Prices')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.legend()
ax.grid(True)
st.pyplot(fig)
plt.close()

# Create signals DataFrame
signals_df = pd.DataFrame(index=stock_data.index)
signals_df['Close'] = stock_data['Close']
signals_df['Signal'] = (best_rf.predict(X) > X['Close']).astype(int)

# Backtest with vectorbt
try:
    # Create portfolio with single column
    portfolio = vbt.Portfolio.from_signals(
        close=signals_df['Close'],
        entries=signals_df['Signal'] == 1,
        exits=signals_df['Signal'] == 0,
        freq='D',
        init_cash=10000,  # Initial investment amount
        fees=0.001  # Commission rate
    )

    # Display backtest results
    st.subheader('Backtest Results')
    
    # Get portfolio stats
    total_return = portfolio.total_return()
    sharpe_ratio = portfolio.sharpe_ratio()
    max_drawdown = portfolio.max_drawdown()
    total_trades = portfolio.count_trades()

    st.write("Total Return: {:.2f}%".format(total_return * 100))
    st.write("Sharpe Ratio: {:.2f}".format(sharpe_ratio))
    st.write("Maximum Drawdown: {:.2f}%".format(max_drawdown * 100))
    st.write("Total Trades: {}".format(total_trades))

    # Create custom portfolio visualization
    fig, ax = plt.subplots(figsize=(14, 7))
    portfolio.plot_cum_returns(ax=ax)
    ax.set_title('Cumulative Returns')
    ax.set_xlabel('Date')
    ax.set_ylabel('Returns')
    st.pyplot(fig)
    plt.close()

except Exception as e:
    st.error(f"Error running backtest: {str(e)}")

# Alpaca API credentials
ALPACA_API_KEY = st.secrets["global"]["ALPACA_API_KEY"]
ALPACA_SECRET_KEY = st.secrets["global"]["ALPACA_SECRET_KEY"]
ALPACA_BASE_URL = 'https://paper-api.alpaca.markets'

# Initialize Alpaca REST API
alpaca = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, api_version='v2')

# Define the trading function
def trade(symbol, prediction, current_price, quantity):
    try:
        if prediction > current_price:
            # Buy signal
            alpaca.submit_order(
                symbol=symbol,
                qty=quantity,
                side='buy',
                type='market',
                time_in_force='day'
            )
            return f"Buying {quantity} shares of {symbol} at {current_price:.2f}"
        else:
            # Sell signal
            alpaca.submit_order(
                symbol=symbol,
                qty=quantity,
                side='sell',
                type='market',
                time_in_force='day'
            )
            return f"Selling {quantity} shares of {symbol} at {current_price:.2f}"
    except Exception as e:
        return f"Error executing trade: {str(e)}"

# Execute trade button
if st.button('Execute Trade'):
    last_prediction = best_rf.predict([X.iloc[-1]])[0]
    last_price = X.iloc[-1]['Close']
    result = trade(symbol, last_prediction, last_price, trade_quantity)
    st.write(result)