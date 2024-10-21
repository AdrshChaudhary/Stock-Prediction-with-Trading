#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import vectorbt as vbt
import matplotlib.pyplot as plt
from alpaca_trade_api.rest import REST
import plotly.graph_objects as go
from datetime import datetime, timedelta

class StockPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        
    def prepare_features(self, data):
        """Prepare features for the model including technical indicators."""
        df = data.copy()
        
        # Basic features
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['MA200'] = df['Close'].rolling(window=200).mean()
        
        # Additional technical indicators
        df['RSI'] = self._calculate_rsi(df['Close'])
        df['MACD'], df['Signal_Line'] = self._calculate_macd(df['Close'])
        df['Volatility'] = df['Close'].rolling(window=20).std()
        
        return df
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD and Signal Line."""
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line

class TradingApp:
    def __init__(self):
        self.predictor = StockPredictor()
        self.setup_streamlit()
        
    def setup_streamlit(self):
        """Setup Streamlit UI components."""
        st.set_page_config(page_title="Advanced Stock Prediction & Trading", layout="wide")
        st.title('Advanced Stock Prediction and Trading Platform')
        
        # Sidebar for user inputs
        with st.sidebar:
            st.header("Configuration")
            self.symbol = st.text_input("Stock Symbol:", "AAPL")
            self.start_date = st.date_input(
                "Start Date:", 
                datetime.now() - timedelta(days=365*2)
            )
            self.end_date = st.date_input("End Date:", datetime.now())
            self.trade_quantity = st.number_input(
                "Trade Quantity:", 
                value=1, 
                min_value=1
            )
    
    @st.cache_data(ttl=3600)
    def load_data(self, symbol, start_date, end_date):
        """Load and cache stock data."""
        try:
            data = yf.download(symbol, start=start_date, end=end_date)
            if data.empty:
                st.error(f"No data found for symbol {symbol}")
                return None
            return data
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None
    
    def train_model(self, X, y):
        """Train and optimize the Random Forest model."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        with st.spinner('Training model... This may take a few minutes.'):
            rf = RandomForestRegressor(random_state=42)
            grid_search = GridSearchCV(
                rf, param_grid, cv=5, 
                scoring='neg_mean_squared_error', n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            
        return grid_search, X_test, y_test
    
    def plot_predictions(self, y_test, predictions):
        """Create an interactive plot of actual vs predicted prices."""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=y_test.index, y=y_test.values,
            mode='markers', name='Actual Prices',
            marker=dict(color='blue', size=8)
        ))
        
        fig.add_trace(go.Scatter(
            x=y_test.index, y=predictions,
            mode='markers', name='Predicted Prices',
            marker=dict(color='red', size=8)
        ))
        
        fig.update_layout(
            title=f'Actual vs Predicted {self.symbol} Prices',
            xaxis_title='Date',
            yaxis_title='Price',
            template='plotly_white',
            hovermode='x unified'
        )
        
        return fig
    
    def run_backtest(self, data, signals):
        """Run and display backtest results."""
        portfolio = vbt.Portfolio.from_signals(
            close=data['Close'],
            entries=signals,
            exits=~signals,
            freq='D',
            init_cash=10000,
            fees=0.001  # 0.1% trading fee
        )
        
        stats = portfolio.stats()
        
        # Display metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Return", f"{stats['total_return']:.2%}")
        with col2:
            st.metric("Sharpe Ratio", f"{stats['sharpe_ratio']:.2f}")
        with col3:
            st.metric("Max Drawdown", f"{stats['max_drawdown']:.2%}")
        with col4:
            st.metric("Total Trades", f"{stats['total_trades']}")
            
        return portfolio
    
    def execute_trade(self, symbol, prediction, current_price):
        """Execute trade through Alpaca API."""
        try:
            alpaca = REST(
                st.secrets["ALPACA_API_KEY"],
                st.secrets["ALPACA_SECRET_KEY"],
                'https://paper-api.alpaca.markets'
            )
            
            side = 'buy' if prediction > current_price else 'sell'
            
            # Check if we have enough buying power or shares to sell
            account = alpaca.get_account()
            if side == 'buy' and float(account.buying_power) < current_price * self.trade_quantity:
                return "Insufficient buying power"
            
            # Submit the order
            order = alpaca.submit_order(
                symbol=symbol,
                qty=self.trade_quantity,
                side=side,
                type='market',
                time_in_force='day'
            )
            
            return f"Successfully placed {side} order for {self.trade_quantity} shares of {symbol}"
            
        except Exception as e:
            return f"Error executing trade: {e}"
    
    def run(self):
        """Main application logic."""
        # Load Data
        data = self.load_data(self.symbol, self.start_date, self.end_date)
        if data is None:
            return
        
        # Prepare features
        processed_data = self.predictor.prepare_features(data)
        processed_data.dropna(inplace=True)
        
        # Create feature matrix
        X = processed_data[['Close', 'MA50', 'MA200', 'RSI', 'MACD', 'Volatility']]
        y = processed_data['Close'].shift(-1)  # Predict next day's close
        
        # Remove the last row since we don't have the next day's price
        X = X[:-1]
        y = y[:-1]
        
        # Train model
        grid_search, X_test, y_test = self.train_model(X, y)
        best_model = grid_search.best_estimator_
        
        # Make predictions
        predictions = best_model.predict(X_test)
        
        # Display model performance
        st.subheader("Model Performance")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Best Model Parameters:", grid_search.best_params_)
        with col2:
            st.write("R² Score:", r2_score(y_test, predictions))
        
        # Plot predictions
        st.plotly_chart(self.plot_predictions(y_test, predictions))
        
        # Run backtest
        signals = best_model.predict(X) > X['Close'].values
        portfolio = self.run_backtest(processed_data, signals)
        
        # Plot portfolio performance
        st.subheader('Portfolio Performance')
        st.plotly_chart(portfolio.plot())
        
        # Trading interface
        st.subheader("Live Trading")
        if st.button('Execute Trade'):
            last_data = X.iloc[-1].values.reshape(1, -1)
            last_prediction = best_model.predict(last_data)[0]
            last_price = X.iloc[-1]['Close']
            
            result = self.execute_trade(self.symbol, last_prediction, last_price)
            st.write(result)

if __name__ == "__main__":
    app = TradingApp()
    app.run()