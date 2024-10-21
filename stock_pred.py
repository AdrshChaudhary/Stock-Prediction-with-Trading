#!/usr/bin/env python
# coding: utf-8

# In[6]:

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import vectorbt as vbt
import matplotlib.pyplot as plt
from alpaca_trade_api.rest import REST, TimeFrame


# In[9]:


# Download stock data
symbol = 'AAPL'
stock_data = yf.download(symbol, start='2020-01-01', end='2024-01-01')

# Calculate moving averages
stock_data['MA50'] = stock_data['Close'].rolling(window=50).mean()
stock_data['MA200'] = stock_data['Close'].rolling(window=200).mean()

# Drop any NaN values
stock_data.dropna(inplace=True)

# Shift the target variable (Close price) to predict the next day
stock_data['Prediction'] = stock_data['Close'].shift(-1)
stock_data.dropna(inplace=True)


# Visualization: Historical Prices with Moving Averages
plt.figure(figsize=(14, 7))
plt.plot(stock_data['Close'], label='Close Price', color='blue')
plt.plot(stock_data['MA50'], label='50-Day MA', color='orange')
plt.plot(stock_data['MA200'], label='200-Day MA', color='red')
plt.title(f'{symbol} Stock Price with Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()


# In[10]:


# Define features and target
X = stock_data[['Close', 'MA50', 'MA200']]
y = stock_data['Prediction']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[11]:


# Define the random forest model
rf = RandomForestRegressor()

# Define the hyperparameters for grid search
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Best model after grid search
best_rf = grid_search.best_estimator_

# Make predictions
predictions = best_rf.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print(f'Best Model Parameters: {grid_search.best_params_}')
print(f'Mean Squared Error: {mse}')


# In[16]:


# Visualization : Actual vs. Predicted Prices (Scatter Plot)
plt.figure(figsize=(14, 7))

# Plotting actual prices
plt.scatter(y_test.index, y_test.values, label='Actual Prices', color='blue', alpha=0.6, edgecolor='k')

# Plotting predicted prices
plt.scatter(y_test.index, predictions, label='Predicted Prices', color='red', alpha=0.6, edgecolor='k')

plt.title(f'Actual vs Predicted {symbol} Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()


# In[19]:


# Create a signal to buy if the model predicts an increase
signals = best_rf.predict(X) > X['Close']

# Backtest with vectorbt
portfolio = vbt.Portfolio.from_signals(
    close=stock_data['Close'],
    entries=signals,
    exits=~signals,
    freq='D'
)

# Display backtest results
print(portfolio.stats())
portfolio.plot().show()


# In[20]:


# Alpaca API credentials
ALPACA_API_KEY = st.secrets["ALPACA_API_KEY"]
ALPACA_SECRET_KEY = st.secrets["ALPACA_SECRET_KEY"]
ALPACA_BASE_URL = 'https://paper-api.alpaca.markets'

# Initialize Alpaca REST API
alpaca = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, api_version='v2')

# Define the trading function
def trade(symbol, prediction, current_price):
    if prediction > current_price:
        # Buy signal
        alpaca.submit_order(
            symbol=symbol,
            qty=1,  # Adjust quantity as needed
            side='buy',
            type='market',
            time_in_force='day'
        )
        print(f"Buying {symbol} at {current_price}")
    else:
        # Sell signal
        alpaca.submit_order(
            symbol=symbol,
            qty=1,  # Adjust quantity as needed
            side='sell',
            type='market',
            time_in_force='day'
        )
        print(f"Selling {symbol} at {current_price}")

# Example usage with the last predicted value
last_prediction = best_rf.predict([X.iloc[-1]])[0]
last_price = X.iloc[-1]['Close']
trade(symbol, last_prediction, last_price)

