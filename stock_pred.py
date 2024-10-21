import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import vectorbt as vbt
import plotly.graph_objects as go
from alpaca_trade_api.rest import REST
from datetime import datetime, timedelta

@st.cache_data
def load_stock_data(_symbol: str, _start_date: str, _end_date: str):
    """Load and cache stock data using string dates."""
    try:
        data = yf.download(_symbol, start=_start_date, end=_end_date)
        if data.empty:
            st.warning(f"No data found for symbol {_symbol}.")
            return None
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

class StockPredictor:
    def __init__(self):
        self.model = None
        
    def prepare_features(self, data):
        """Prepare features for the model including technical indicators."""
        try:
            df = data.copy()
            if df.empty:
                st.error("Data is empty. Cannot prepare features.")
                return None
            
            # Basic features
            df['MA50'] = df['Close'].rolling(window=50).mean()
            df['MA200'] = df['Close'].rolling(window=200).mean()
            
            # Additional technical indicators
            df['RSI'] = self._calculate_rsi(df['Close'])
            df['MACD'], df['Signal_Line'] = self._calculate_macd(df['Close'])
            df['Volatility'] = df['Close'].rolling(window=20).std()
            
            return df
        except Exception as e:
            st.error(f"Error preparing features: {str(e)}")
            return None
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index."""
        if len(prices) < period:
            return pd.Series([np.nan] * len(prices))
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD and Signal Line."""
        if len(prices) < slow:
            return pd.Series([np.nan] * len(prices)), pd.Series([np.nan] * len(prices))
        
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
        
        with st.sidebar:
            st.header("Configuration")
            self.symbol = st.text_input("Stock Symbol:", "AAPL").upper()
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
        try:
            # Convert to numpy arrays if they're not already
            X = np.array(X)
            y = np.array(y)
            
            # Ensure X and y have the same number of samples
            min_len = min(len(X), len(y))
            X = X[:min_len]
            y = y[:min_len]
            
            # Ensure there's enough data for training
            if len(X) < 10:
                st.error("Not enough data to train the model.")
                return None, None, None
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            param_grid = {
                'n_estimators': [100],
                'max_depth': [10],
                'min_samples_split': [2],
                'min_samples_leaf': [1]
            }
            
            with st.spinner('Training model... This may take a few minutes.'):
                rf = RandomForestRegressor(random_state=42)
                grid_search = GridSearchCV(
                    rf, param_grid, cv=3, 
                    scoring='neg_mean_squared_error', n_jobs=-1
                )
                grid_search.fit(X_train, y_train)
                
            return grid_search, X_test, y_test
        except Exception as e:
            st.error(f"Error training model: {str(e)}")
            return None, None, None
    
    def plot_predictions(self, y_test, predictions):
        """Create an interactive plot of actual vs predicted prices."""
        try:
            # Convert to numpy arrays
            y_test_values = np.array(y_test)
            predictions = np.array(predictions)
            
            # Use actual dates from y_test for plotting
            dates = pd.date_range(end=self.end_date, periods=len(y_test), freq='B')  # Use business day frequency
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=y_test_values,
                mode='lines+markers',
                name='Actual Prices',
                line=dict(color='blue', width=1),
                marker=dict(color='blue', size=6)
            ))
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=predictions,
                mode='lines+markers',
                name='Predicted Prices',
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
        except Exception as e:
            st.error(f"Error plotting predictions: {str(e)}")
            return None

    def run_backtest(self, data, signals):
        """Run and display backtest results with improved error handling and calculations."""
        try:
            # Create a clean DataFrame with required data
            backtest_df = pd.DataFrame({
                'Close': data['Close'],
                'Signal': signals
            }).dropna()  # Remove any NaN values
            
            if backtest_df.empty:
                st.error("No valid signals for backtesting.")
                return None
            
            # Ensure signals are boolean
            entry_signals = backtest_df['Signal'].astype(bool)
            exit_signals = ~entry_signals  # Inverse of entry signals
            
            # Initialize portfolio with proper parameters
            portfolio = vbt.Portfolio.from_signals(
                close=backtest_df['Close'],
                entries=entry_signals,
                exits=exit_signals,
                freq='1D',  # Explicitly set frequency
                init_cash=10000,
                fees=0.001,
                sl_stop=0.15,  # Add stop loss at 15%
                tp_stop=0.30   # Add take profit at 30%
            )
            
            # Calculate metrics with safe error handling
            metrics = {}
            
            # Total Return
            try:
                final_value = portfolio.final_value()
                init_cash = portfolio.init_cash
                metrics['total_return'] = ((final_value - init_cash) / init_cash) * 100
            except Exception as e:
                st.warning(f"Could not calculate total return: {str(e)}")
                metrics['total_return'] = 0.0
                
            # Sharpe Ratio
            try:
                metrics['sharpe_ratio'] = portfolio.sharpe_ratio(risk_free=0.02)
            except Exception as e:
                st.warning(f"Could not calculate Sharpe ratio: {str(e)}")
                metrics['sharpe_ratio'] = 0.0
                
            # Max Drawdown
            try:
                metrics['max_drawdown'] = portfolio.max_drawdown() * 100
            except Exception as e:
                st.warning(f"Could not calculate max drawdown: {str(e)}")
                metrics['max_drawdown'] = 0.0
                
            # Trade Statistics
            try:
                trades = portfolio.trades
                metrics['total_trades'] = len(trades.records)
                metrics['win_rate'] = (trades['Profit'].values > 0).mean() * 100
            except Exception as e:
                st.warning(f"Could not calculate trades statistics: {str(e)}")
                metrics['total_trades'] = 0
                metrics['win_rate'] = 0.0
            
            # Display metrics
            for key, value in metrics.items():
                st.write(f"{key.replace('_', ' ').title()}: {value:.2f}")
            
            return portfolio
        except Exception as e:
            st.error(f"Error running backtest: {str(e)}")
            return None

    def execute_trade(self, action: str, quantity: int):
        """Execute trades using the Alpaca API."""
        try:
            # Check Alpaca keys
            ALPACA_API_KEY = st.secrets['alpaca_api_key']
            ALPACA_SECRET_KEY = st.secrets['alpaca_secret_key']
            ALPACA_BASE_URL = st.secrets['alpaca_base_url']
            
            if not all([ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL]):
                st.error("Alpaca API keys not configured.")
                return
            
            # Initialize API
            api = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url=ALPACA_BASE_URL)
            current_price = api.get_last_trade(self.symbol).price
            
            # Execute the trade
            if action == 'Buy':
                api.submit_order(
                    symbol=self.symbol,
                    qty=quantity,
                    side='buy',
                    type='market',
                    time_in_force='gtc'
                )
                st.success(f"Successfully bought {quantity} shares of {self.symbol} at ${current_price:.2f}.")
            elif action == 'Sell':
                api.submit_order(
                    symbol=self.symbol,
                    qty=quantity,
                    side='sell',
                    type='market',
                    time_in_force='gtc'
                )
                st.success(f"Successfully sold {quantity} shares of {self.symbol} at ${current_price:.2f}.")
        except Exception as e:
            st.error(f"Error executing trade: {str(e)}")

if __name__ == "__main__":
    app = TradingApp()
    
    # Load data
    stock_data = app.load_data(app.symbol, app.start_date, app.end_date)
    if stock_data is not None:
        features = app.predictor.prepare_features(stock_data)
        
        if features is not None:
            # Prepare data for training
            X = features[['MA50', 'MA200', 'RSI', 'MACD', 'Signal_Line', 'Volatility']].dropna()
            y = features['Close'].shift(-1).dropna()  # Predict next day's closing price
            if not X.empty and not y.empty:
                model, X_test, y_test = app.train_model(X, y)
                if model is not None:
                    predictions = model.predict(X_test)
                    # Plotting results
                    fig = app.plot_predictions(y_test, predictions)
                    if fig:
                        st.plotly_chart(fig)
                        
            # Example trade execution
            if st.button("Execute Buy"):
                app.execute_trade("Buy", app.trade_quantity)
            if st.button("Execute Sell"):
                app.execute_trade("Sell", app.trade_quantity)
            
            # Backtesting
            signals = (predictions > y_test)  # Example buy signal if predicted > actual
            portfolio = app.run_backtest(stock_data, signals)
