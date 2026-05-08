# 💎 Ultra Stock Intelligence Pro

An end-to-end stock market analysis and prediction system. This project transitions from a deep-dive research notebook to a high-performance interactive dashboard, providing traders and analysts with AI-driven insights.

## 🚀 Features

- **AI-Powered Predictions**: Uses Random Forest and Linear Regression to predict next-day closing prices with accuracy metrics (R², MAE).
- **Interactive Technical Analysis**: Real-time Candlestick charts with Bollinger Bands, RSI, and Moving Averages.
- **Fundamental Insights**: Direct access to Market Cap, P/E Ratios, Dividend Yields, and more via `yfinance`.
- **Live Intelligence News**: Real-time news feed integration for the selected asset.
- **Strategy Backtesting**: Performance comparison between AI strategy and traditional Buy & Hold.
- **Automated Risk Management**: Real-time calculation of Stop-Loss and Take-Profit levels.
- **Educational Suite**: Built-in documentation explaining the mathematics behind the indicators and models.

## 🛠️ Tech Stack

- **Language**: Python 3.10+
- **Framework**: [Streamlit](https://streamlit.io/) (Interactive Dashboard)
- **Visualization**: [Plotly](https://plotly.com/python/) (Dynamic Charts)
- **Machine Learning**: Scikit-Learn
- **Data Source**: Yahoo Finance API (`yfinance`)
- **Data Manipulation**: Pandas, NumPy

## 📦 Installation & Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/habibaasaber/Stock-Market-Analysis.git
   cd Stock-Market-Analysis
   ```

2. **Install dependencies**:
   ```bash
   pip install streamlit yfinance pandas numpy plotly scikit-learn
   ```

3. **Run the Dashboard**:
   ```bash
   streamlit run stock_dashboard.py
   ```

## 🧠 Methodology

### Data Pipeline
The system fetches historical data, applies automated cleaning (handling NaNs, outliers, and volume spikes), and performs feature engineering to generate signals like RSI and Volatility.

### Prediction Engine
The AI Engine trains on historical features and evaluates performance on a non-shuffled test set to ensure zero data leakage. It currently supports:
- **Random Forest**: For capturing non-linear market patterns.
- **Linear Regression**: For stable trend analysis.

---
Built with ❤️ by Habiba Saber
