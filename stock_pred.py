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

@st.cache_data
def load_stock_data(_symbol: str, _start_date: str, _end_date: str):
    """Load and cache stock data using string dates."""
    try:
        data = yf.download(_symbol, start=_start_date, end=_end_date)
        if data.empty:
            return None
        return data
    except Exception as e:
        return None

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
    
    def load_data(self, symbol, start_date, end_date):
        """Wrapper for loading data that handles date conversion."""
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        return load_stock_data(symbol, start_date_str, end_date_str)
    
    def train_model(self, X, y):
        """Train and optimize the Random Forest model."""
        # Ensure X and y have the same number of samples
        min_len = min(len(X), len(y))
        X = X[:min_len]
        y = y[:min_len]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        with st.spinner('Training model... This may take a few minutes.'):
            rf = RandomForestRegressor(random_state=42)
            grid_search = GridSearchCV(
                rf, param_grid, cv=3, 
                scoring='neg_mean_squared_error', n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            
        return grid_search, X_test, y_test
    
    def plot_predictions(self, y_test, predictions):
        """Create an interactive plot of actual vs predicted prices."""
        # Ensure we're working with numpy arrays
        y_test_values = np.array(y_test)
        predictions = np.array(predictions)
        
        # Create a date index for plotting
        dates = pd.date_range(end=self.end_date, periods=len(y_test), freq='D')
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dates, y=y_test_values,
            mode='lines+markers', name='Actual Prices',
            line=dict(color='blue', width=1),
            marker=dict(color='blue', size=6)
        ))
        
        fig.add_trace(go.Scatter(
            x=dates, y=predictions,
            mode='lines+markers', name='Predicted Prices',
            line=dict(color='red', width=1),
            marker=dict(color='red', size=6)
        ))
        
        fig.update_layout(
            title=f'Actual vs Predicted {self.symbol} Prices',
            xaxis_title='Date',
            yaxis_title='Price',
            template='plotly_white',
            hovermode='x unified',
            showlegend=True
        )
        
        return fig
    
    def run_backtest(self, data, signals):
        """Run and display backtest results."""
        # Ensure signals array matches data length
        signals = signals[:len(data)]
        
        portfolio = vbt.Portfolio.from_signals(
            close=data['Close'],
            entries=signals,
            exits=~signals,
            freq='D',
            init_cash=10000,
            fees=0.001
        )
        
        # Calculate metrics manually from portfolio results
        total_return = (portfolio.final_value - portfolio.init_cash) / portfolio.init_cash
        sharpe_ratio = portfolio.sharpe_ratio()
        max_drawdown = portfolio.max_drawdown()
        total_trades = len(portfolio.trades)
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Return", f"{total_return:.2%}")
        with col2:
            st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
        with col3:
            st.metric("Max Drawdown", f"{max_drawdown:.2%}")
        with col4:
            st.metric("Total Trades", f"{total_trades}")
            
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
            
            account = alpaca.get_account()
            if side == 'buy' and float(account.buying_power) < current_price * self.trade_quantity:
                return "Insufficient buying power"
            
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
        data = self.load_data(self.symbol, self.start_date, self.end_date)
        if data is None:
            st.error(f"No data found for symbol {self.symbol}")
            return
        
        processed_data = self.predictor.prepare_features(data)
        processed_data.dropna(inplace=True)
        
        if len(processed_data) < 200:
            st.error("Not enough data for analysis. Please select a longer date range.")
            return
            
        # Create feature matrix and target
        X = processed_data[['Close', 'MA50', 'MA200', 'RSI', 'MACD', 'Volatility']].values
        y = processed_data['Close'].shift(-1).values[:-1]  # Remove last row
        X = X[:-1]  # Remove last row to match y length
        
        # Train model and get predictions
        grid_search, X_test, y_test = self.train_model(X, y)
        best_model = grid_search.best_estimator_
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
        signals = best_model.predict(X) > X[:, 0]  # Compare with closing prices
        portfolio = self.run_backtest(processed_data[:-1], signals)
        
        # Plot portfolio performance
        st.subheader('Portfolio Performance')
        st.plotly_chart(portfolio.plot())
        
        # Trading interface
        st.subheader("Live Trading")
        if st.button('Execute Trade'):
            last_data = X[-1].reshape(1, -1)
            last_prediction = best_model.predict(last_data)[0]
            last_price = X[-1, 0]  # Close price
            
            result = self.execute_trade(self.symbol, last_prediction, last_price)
            st.write(result)

if __name__ == "__main__":
    app = TradingApp()
    app.run()