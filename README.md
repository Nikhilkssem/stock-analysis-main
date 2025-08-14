# 📈 Stock Price Analysis Dashboard

A modern, interactive Streamlit dashboard for analyzing and forecasting stock price data with ARIMA and Exponential Smoothing models, advanced visualizations, and user-friendly UI.

## Features
- Upload your own stock CSV or use sample data
- Interactive time series plots (Plotly)
- Monthly, rolling, and seasonal analysis
- ACF/PACF, stationarity, and decomposition tools
- ARIMA and Exponential Smoothing forecasting
- Model metrics (MAE, RMSE, R², AIC, BIC)
- Beautiful, responsive UI

## Setup
1. **Clone the repo**
   ```bash
   git clone <repo-url>
   cd stock-analysis-main
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the app**
   ```bash
   streamlit run stock_analysis.py
   ```

## Usage
- Use the sidebar to upload a CSV (with Date, Open, High, Low, Close, Volume columns) or generate sample data.
- Adjust ARIMA parameters and test size in the sidebar.
- Explore statistics, trends, decomposition, and model forecasts in the main view.

## Deployment
- Deploy on [Streamlit Community Cloud](https://streamlit.io/cloud) or any cloud platform supporting Python 3.9+.
- For Docker, create a Dockerfile using the requirements.txt.

## File Structure
- `stock_analysis.py` — Main Streamlit app
- `stock_data.csv` — Example data file
- `visuals/` — Saved plots (auto-generated)

## License
MIT 
