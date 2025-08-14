import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Stock Price Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern CSS for better styling and dark mode support
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .stAlert {
        margin-top: 1rem;
    }
    /* Responsive tweaks */
    @media (max-width: 900px) {
        .main-header { font-size: 2rem; }
    }
</style>
""", unsafe_allow_html=True)

class StockAnalyzer:
    def __init__(self):
        self.df = None
        self.df_resampled = None
        self.train = None
        self.test = None
        self.model_fit = None
        self.forecast = None
        
    def load_data(self, uploaded_file):
        """Load and preprocess the stock data. Returns True if successful."""
        try:
            if uploaded_file is not None:
                self.df = pd.read_csv(uploaded_file, parse_dates=True, index_col='Date')
                
                # Drop unnamed columns
                unnamed_cols = [col for col in self.df.columns if 'Unnamed' in col]
                if unnamed_cols:
                    self.df.drop(columns=unnamed_cols, inplace=True)
                
                # Ensure numeric columns
                numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                for col in numeric_columns:
                    if col in self.df.columns:
                        self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                
                # Remove any rows with all NaN values
                self.df.dropna(how='all', inplace=True)
                
                return True
            return False
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return False
    
    def create_sample_data(self):
        """Create sample stock data for demonstration purposes."""
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='D')
        
        # Generate realistic stock price data with trend and seasonality
        trend = np.linspace(100, 150, len(dates))
        seasonal = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
        noise = np.random.normal(0, 5, len(dates))
        
        prices = trend + seasonal + noise
        volumes = np.random.lognormal(15, 1, len(dates))
        
        # Create OHLC data
        opens = prices + np.random.normal(0, 1, len(dates))
        highs = np.maximum(opens, prices) + np.abs(np.random.normal(0, 2, len(dates)))
        lows = np.minimum(opens, prices) - np.abs(np.random.normal(0, 2, len(dates)))
        closes = prices
        
        self.df = pd.DataFrame({
            'Open': opens,
            'High': highs,
            'Low': lows,
            'Close': closes,
            'Volume': volumes
        }, index=dates)
        
        return True
    
    def basic_statistics(self):
        """Calculate and return basic statistics as a dictionary."""
        if self.df is None:
            return None
            
        stats = {
            'Total Trading Days': len(self.df),
            'Date Range': f"{self.df.index.min().strftime('%Y-%m-%d')} to {self.df.index.max().strftime('%Y-%m-%d')}",
            'Average High Price': f"${self.df['High'].mean():.2f}",
            'Maximum High Price': f"${self.df['High'].max():.2f}",
            'Minimum High Price': f"${self.df['High'].min():.2f}",
            'Average Daily Volume': f"{self.df['Volume'].mean():,.0f}" if 'Volume' in self.df.columns else "N/A"
        }
        return stats
    
    def plot_price_trends(self):
        """Create interactive price trend plots using Plotly."""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Stock Prices Over Time', 'Volume Over Time'),
            vertical_spacing=0.1,
            specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
        )
        
        # Price plot
        fig.add_trace(
            go.Scatter(x=self.df.index, y=self.df['High'], name='High', line=dict(color='red')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=self.df.index, y=self.df['Low'], name='Low', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=self.df.index, y=self.df['Close'], name='Close', line=dict(color='green')),
            row=1, col=1
        )
        
        # Volume plot
        if 'Volume' in self.df.columns:
            fig.add_trace(
                go.Scatter(x=self.df.index, y=self.df['Volume'], name='Volume', 
                          line=dict(color='purple'), fill='tonexty'),
                row=2, col=1
            )
        
        fig.update_layout(
            title="Stock Price Analysis",
            height=700,
            showlegend=True,
            hovermode='x unified'
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        return fig
    
    def plot_monthly_resampling(self):
        """Create monthly resampled data visualization."""
        self.df_resampled = self.df.resample('M').mean(numeric_only=True)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=self.df_resampled.index,
            y=self.df_resampled['High'],
            mode='lines+markers',
            name='Monthly Avg High',
            line=dict(color='blue', width=3)
        ))
        
        fig.update_layout(
            title="Monthly Average High Prices",
            xaxis_title="Date",
            yaxis_title="Average High Price ($)",
            height=500,
            hovermode='x'
        )
        
        return fig
    
    def plot_acf_pacf(self, column='Volume', lags=30):
        """Create ACF and PACF plots for a given column."""
        if column not in self.df.columns:
            return None
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # ACF plot
        plot_acf(self.df[column].dropna(), lags=lags, ax=ax1)
        ax1.set_title(f'Autocorrelation Function - {column}')
        
        # PACF plot
        plot_pacf(self.df[column].dropna(), lags=lags, ax=ax2)
        ax2.set_title(f'Partial Autocorrelation Function - {column}')
        
        plt.tight_layout()
        return fig
    
    def analyze_stationarity(self):
        """Perform stationarity analysis and return plotly figure and ADF results."""
        # Create differenced series
        self.df['high_diff'] = self.df['High'].diff()
        
        # ADF test on original series
        adf_original = adfuller(self.df['High'].dropna())
        
        # ADF test on differenced series
        adf_diff = adfuller(self.df['high_diff'].dropna())
        
        # Rolling statistics
        rolling_mean = self.df['High'].rolling(window=30).mean()
        rolling_std = self.df['High'].rolling(window=30).std()
        
        # Create visualization
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Original vs Differenced Series', 'Rolling Mean & Std', 'Distribution Comparison'),
            vertical_spacing=0.1
        )
        
        # Original vs Differenced
        fig.add_trace(
            go.Scatter(x=self.df.index, y=self.df['High'], name='Original High', line=dict(color='red')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=self.df.index, y=self.df['high_diff'], name='Differenced High', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Rolling statistics
        fig.add_trace(
            go.Scatter(x=self.df.index, y=self.df['High'], name='Original', line=dict(color='black')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=rolling_mean.index, y=rolling_mean, name='Rolling Mean', line=dict(color='green')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=rolling_std.index, y=rolling_std, name='Rolling Std', line=dict(color='orange')),
            row=2, col=1
        )
        
        fig.update_layout(height=900, showlegend=True)
        
        # Return results
        results = {
            'ADF Original': {
                'Statistic': adf_original[0],
                'p-value': adf_original[1],
                'Is Stationary': adf_original[1] < 0.05
            },
            'ADF Differenced': {
                'Statistic': adf_diff[0],
                'p-value': adf_diff[1],
                'Is Stationary': adf_diff[1] < 0.05
            }
        }
        
        return fig, results
    
    def seasonal_decomposition(self):
        """Perform seasonal decomposition and return a plotly figure."""
        # Use a more appropriate period based on data frequency
        if len(self.df) > 365:
            period = min(365, len(self.df) // 4)  # Annual or quarterly
        else:
            period = min(30, len(self.df) // 4)   # Monthly
            
        decomposition = seasonal_decompose(
            self.df['High'].dropna(), 
            model='additive', 
            period=period
        )
        
        # Create interactive plot
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=('Original', 'Trend', 'Seasonal', 'Residual'),
            vertical_spacing=0.05
        )
        
        fig.add_trace(
            go.Scatter(x=decomposition.observed.index, y=decomposition.observed, name='Original'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=decomposition.trend.index, y=decomposition.trend, name='Trend'),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, name='Seasonal'),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=decomposition.resid.index, y=decomposition.resid, name='Residual'),
            row=4, col=1
        )
        
        fig.update_layout(
            height=800,
            title="Seasonal Decomposition of High Prices",
            showlegend=False
        )
        
        return fig
    
    def fit_arima_model(self, order=(1, 1, 1), test_size=0.2):
        """Fit ARIMA model and generate forecasts. Returns plot, metrics, and summary."""
        # Train-test split
        train_size = int(len(self.df) * (1 - test_size))
        self.train = self.df['High'].iloc[:train_size]
        self.test = self.df['High'].iloc[train_size:]
        
        # Fit ARIMA model
        try:
            self.model_fit = ARIMA(self.train, order=order).fit()
            
            # Generate forecasts
            self.forecast = self.model_fit.forecast(steps=len(self.test))
            
            # Calculate metrics
            mae = mean_absolute_error(self.test, self.forecast)
            rmse = np.sqrt(mean_squared_error(self.test, self.forecast))
            r2 = r2_score(self.test, self.forecast)
            
            # Create forecast plot
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=self.train.index, y=self.train,
                name='Training Data', line=dict(color='blue')
            ))
            
            fig.add_trace(go.Scatter(
                x=self.test.index, y=self.test,
                name='Actual Test Data', line=dict(color='green')
            ))
            
            fig.add_trace(go.Scatter(
                x=self.test.index, y=self.forecast,
                name='Forecast', line=dict(color='red', dash='dash')
            ))
            
            fig.update_layout(
                title=f"ARIMA{order} Forecast vs Actual High Prices",
                xaxis_title="Date",
                yaxis_title="High Price ($)",
                height=500,
                hovermode='x'
            )
            
            metrics = {
                'MAE': mae,
                'RMSE': rmse,
                'R²': r2,
                'AIC': self.model_fit.aic,
                'BIC': self.model_fit.bic
            }
            
            return fig, metrics, self.model_fit.summary()
            
        except Exception as e:
            st.error(f"Error fitting ARIMA model: {str(e)}")
            return None, None, None
    
    def fit_exponential_smoothing(self, test_size=0.2):
        """Fit Exponential Smoothing model as alternative. Returns plot and metrics."""
        train_size = int(len(self.df) * (1 - test_size))
        train = self.df['High'].iloc[:train_size]
        test = self.df['High'].iloc[train_size:]
        
        try:
            # Fit Exponential Smoothing
            model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=30)
            model_fit = model.fit()
            
            # Generate forecasts
            forecast = model_fit.forecast(steps=len(test))
            
            # Calculate metrics
            mae = mean_absolute_error(test, forecast)
            rmse = np.sqrt(mean_squared_error(test, forecast))
            r2 = r2_score(test, forecast)
            
            # Create forecast plot
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=train.index, y=train,
                name='Training Data', line=dict(color='blue')
            ))
            
            fig.add_trace(go.Scatter(
                x=test.index, y=test,
                name='Actual Test Data', line=dict(color='green')
            ))
            
            fig.add_trace(go.Scatter(
                x=test.index, y=forecast,
                name='ES Forecast', line=dict(color='orange', dash='dash')
            ))
            
            fig.update_layout(
                title="Exponential Smoothing Forecast vs Actual High Prices",
                xaxis_title="Date",
                yaxis_title="High Price ($)",
                height=500,
                hovermode='x'
            )
            
            metrics = {
                'MAE': mae,
                'RMSE': rmse,
                'R²': r2
            }
            
            return fig, metrics
            
        except Exception as e:
            st.error(f"Error fitting Exponential Smoothing model: {str(e)}")
            return None, None

# Main Streamlit App
def main():
    st.markdown('<h1 class="main-header">📈 Stock Price Analysis Dashboard</h1>', unsafe_allow_html=True)
    st.info("""
    **Instructions:**
    - Use the sidebar to upload your own stock CSV or generate sample data.
    - Adjust ARIMA parameters and test size for forecasting.
    - Explore statistics, trends, decomposition, and model forecasts below.
    """, icon="ℹ️")
    
    # Initialize analyzer
    analyzer = StockAnalyzer()
    
    # Sidebar for controls
    st.sidebar.header("Configuration")
    st.sidebar.markdown("""
    <small>Upload a CSV with columns: <b>Date, Open, High, Low, Close, Volume</b>.<br>
    Or use sample data for demo.</small>
    """, unsafe_allow_html=True)
    
    data_option = st.sidebar.radio(
        "Choose data source:",
        ["Upload CSV file", "Use sample data"]
    )
    
    if data_option == "Upload CSV file":
        uploaded_file = st.sidebar.file_uploader(
            "Choose a CSV file",
            type="csv",
            help="Upload a CSV file with Date, Open, High, Low, Close, Volume columns"
        )
        
        if uploaded_file is not None:
            if analyzer.load_data(uploaded_file):
                st.sidebar.success("Data loaded successfully!")
            else:
                st.sidebar.error("Failed to load data")
                return
        else:
            st.warning("Please upload a CSV file to continue.")
            return
    else:
        if analyzer.create_sample_data():
            st.sidebar.success("Sample data generated!")
    
    # Analysis parameters
    st.sidebar.subheader("Analysis Parameters")
    arima_p = st.sidebar.slider("ARIMA p (AR order)", 0, 5, 1, help="Autoregressive order")
    arima_d = st.sidebar.slider("ARIMA d (Differencing)", 0, 2, 1, help="Number of differences")
    arima_q = st.sidebar.slider("ARIMA q (MA order)", 0, 5, 1, help="Moving average order")
    test_size = st.sidebar.slider("Test size ratio", 0.1, 0.4, 0.2, help="Fraction of data for testing")
    
    # Main content
    if analyzer.df is not None:
        # Basic Statistics
        st.header("📊 Basic Statistics")
        stats = analyzer.basic_statistics()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Trading Days", stats['Total Trading Days'])
            st.metric("Average High Price", stats['Average High Price'])
        with col2:
            st.metric("Maximum High Price", stats['Maximum High Price'])
            st.metric("Minimum High Price", stats['Minimum High Price'])
        with col3:
            st.metric("Date Range", stats['Date Range'])
            st.metric("Average Daily Volume", stats['Average Daily Volume'])
        
        # Data preview
        st.subheader("Data Preview")
        st.dataframe(analyzer.df.head(10), use_container_width=True)
        
        # Price Trends
        st.header("📈 Price Trends")
        price_fig = analyzer.plot_price_trends()
        st.plotly_chart(price_fig, use_container_width=True)
        
        # Monthly Resampling
        st.header("📅 Monthly Analysis")
        monthly_fig = analyzer.plot_monthly_resampling()
        st.plotly_chart(monthly_fig, use_container_width=True)
        
        # ACF/PACF Analysis
        st.header("🔄 Autocorrelation Analysis")
        acf_column = st.selectbox("Select column for ACF/PACF analysis:", 
                                 [col for col in analyzer.df.columns if col != 'high_diff'],
                                 help="Choose a numeric column for autocorrelation analysis.")
        
        if acf_column:
            acf_fig = analyzer.plot_acf_pacf(acf_column)
            if acf_fig:
                st.pyplot(acf_fig)
        
        # Stationarity Analysis
        st.header("📊 Stationarity Analysis")
        stationarity_fig, adf_results = analyzer.analyze_stationarity()
        st.plotly_chart(stationarity_fig, use_container_width=True)
        
        # Display ADF test results
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Series ADF Test")
            st.write(f"**Statistic:** {adf_results['ADF Original']['Statistic']:.4f}")
            st.write(f"**p-value:** {adf_results['ADF Original']['p-value']:.4f}")
            if adf_results['ADF Original']['Is Stationary']:
                st.success("Series is stationary")
            else:
                st.warning("Series is non-stationary")
        
        with col2:
            st.subheader("Differenced Series ADF Test")
            st.write(f"**Statistic:** {adf_results['ADF Differenced']['Statistic']:.4f}")
            st.write(f"**p-value:** {adf_results['ADF Differenced']['p-value']:.4f}")
            if adf_results['ADF Differenced']['Is Stationary']:
                st.success("Series is stationary")
            else:
                st.warning("Series is non-stationary")
        
        # Seasonal Decomposition
        st.header("🔄 Seasonal Decomposition")
        seasonal_fig = analyzer.seasonal_decomposition()
        st.plotly_chart(seasonal_fig, use_container_width=True)
        
        # Model Fitting and Forecasting
        st.header("🤖 Model Fitting & Forecasting")
        
        model_tabs = st.tabs(["ARIMA Model", "Exponential Smoothing"])
        
        with model_tabs[0]:
            st.subheader(f"ARIMA({arima_p},{arima_d},{arima_q}) Model")
            
            if st.button("Fit ARIMA Model", type="primary"):
                with st.spinner("Fitting ARIMA model..."):
                    arima_fig, arima_metrics, arima_summary = analyzer.fit_arima_model(
                        order=(arima_p, arima_d, arima_q), 
                        test_size=test_size
                    )
                
                if arima_fig:
                    st.plotly_chart(arima_fig, use_container_width=True)
                    
                    # Display metrics
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric("MAE", f"{arima_metrics['MAE']:.2f}")
                    with col2:
                        st.metric("RMSE", f"{arima_metrics['RMSE']:.2f}")
                    with col3:
                        st.metric("R²", f"{arima_metrics['R²']:.3f}")
                    with col4:
                        st.metric("AIC", f"{arima_metrics['AIC']:.2f}")
                    with col5:
                        st.metric("BIC", f"{arima_metrics['BIC']:.2f}")
                    
                    # Model summary
                    with st.expander("Model Summary"):
                        st.text(str(arima_summary))
        
        with model_tabs[1]:
            st.subheader("Exponential Smoothing Model")
            
            if st.button("Fit Exponential Smoothing Model", type="primary"):
                with st.spinner("Fitting Exponential Smoothing model..."):
                    es_fig, es_metrics = analyzer.fit_exponential_smoothing(test_size=test_size)
                
                if es_fig:
                    st.plotly_chart(es_fig, use_container_width=True)
                    
                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("MAE", f"{es_metrics['MAE']:.2f}")
                    with col2:
                        st.metric("RMSE", f"{es_metrics['RMSE']:.2f}")
                    with col3:
                        st.metric("R²", f"{es_metrics['R²']:.3f}")

if __name__ == "__main__":
    main()