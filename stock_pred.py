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
            
            # Create a date index for plotting
            dates = pd.date_range(end=self.end_date, periods=len(y_test), freq='D')
            
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
        })
        backtest_df = backtest_df.dropna()  # Remove any NaN values
        
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
            metrics['win_rate'] = (trades.win_rate * 100) if trades.win_rate is not None else 0.0
        except Exception as e:
            st.warning(f"Could not calculate trade statistics: {str(e)}")
            metrics['total_trades'] = 0
            metrics['win_rate'] = 0.0
        
        # Display metrics in columns
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Return", f"{metrics['total_return']:.2f}%")
        with col2:
            st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
        with col3:
            st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2f}%")
        with col4:
            st.metric("Total Trades", f"{metrics['total_trades']}")
        with col5:
            st.metric("Win Rate", f"{metrics['win_rate']:.1f}%")
        
        # Plot cumulative returns
        try:
            returns = portfolio.returns()
            cumulative_returns = (returns + 1).cumprod() - 1
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=cumulative_returns.index,
                y=cumulative_returns.values * 100,  # Convert to percentage
                mode='lines',
                name='Portfolio Returns',
                line=dict(color='blue', width=2)
            ))
            
            # Add buy/sell markers
            entry_points = backtest_df[entry_signals]['Close']
            exit_points = backtest_df[exit_signals]['Close']
            
            fig.add_trace(go.Scatter(
                x=entry_points.index,
                y=[0] * len(entry_points),  # Plot at 0% level
                mode='markers',
                name='Buy Signals',
                marker=dict(symbol='triangle-up', size=10, color='green')
            ))
            
            fig.add_trace(go.Scatter(
                x=exit_points.index,
                y=[0] * len(exit_points),  # Plot at 0% level
                mode='markers',
                name='Sell Signals',
                marker=dict(symbol='triangle-down', size=10, color='red')
            ))
            
            fig.update_layout(
                title='Portfolio Performance',
                xaxis_title='Date',
                yaxis_title='Cumulative Return (%)',
                template='plotly_white',
                showlegend=True,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error creating performance chart: {str(e)}")
        
        return portfolio
        
    except Exception as e:
        st.error(f"Error in backtesting: {str(e)}")
        return None
    
    def execute_trade(self, symbol, prediction, current_price):
        """Execute trade through Alpaca API."""
        try:
            alpaca = REST(
                st.secrets["global"]["ALPACA_API_KEY"],
                st.secrets["global"]["ALPACA_SECRET_KEY"],
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
            return f"Error executing trade: {str(e)}"
    
    def run(self):
        """Main application logic."""
        try:
            # Load Data
            data = self.load_data(self.symbol, self.start_date, self.end_date)
            if data is None:
                st.error(f"No data found for symbol {self.symbol}")
                return
            
            # Prepare features
            processed_data = self.predictor.prepare_features(data)
            if processed_data is None:
                return
                
            processed_data.dropna(inplace=True)
            
            if len(processed_data) < 200:
                st.error("Not enough data for analysis. Please select a longer date range.")
                return
                
            # Create feature matrix and target
            feature_cols = ['Close', 'MA50', 'MA200', 'RSI', 'MACD', 'Volatility']
            X = processed_data[feature_cols].values
            y = processed_data['Close'].shift(-1).values[:-1]  # Remove last row
            X = X[:-1]  # Remove last row to match y length
            
            # Train model and get predictions
            grid_search, X_test, y_test = self.train_model(X, y)
            if grid_search is None:
                return
                
            best_model = grid_search.best_estimator_
            predictions = best_model.predict(X_test)
            
            # Display model performance
            st.subheader("Model Performance")
            col1, col2 = st.columns(2)
            with col1:
                st.write("Best Model Parameters:", grid_search.best_params_)
            with col2:
                st.write("R² Score:", f"{r2_score(y_test, predictions):.4f}")
            
            # Plot predictions
            pred_plot = self.plot_predictions(y_test, predictions)
            if pred_plot is not None:
                st.plotly_chart(pred_plot)
            
            # Run backtest
            signals = best_model.predict(X) > X[:, 0]  # Compare with closing prices
            self.run_backtest(processed_data[:-1], signals)
            
            # Trading interface
            st.subheader("Live Trading")
            if st.button('Execute Trade'):
                last_data = X[-1].reshape(1, -1)
                last_prediction = best_model.predict(last_data)[0]
                last_price = X[-1, 0]  # Close price
                
                result = self.execute_trade(self.symbol, last_prediction, last_price)
                st.write(result)
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    app = TradingApp()
    app.run()