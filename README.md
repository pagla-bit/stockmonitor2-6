# 📈 Enhanced Stock Dashboard v2.3 - Modular Edition

A comprehensive, modular stock analysis dashboard built with Streamlit, featuring technical analysis, machine learning predictions, sentiment analysis, and strategy backtesting.

## 🌟 Key Features

### Technical Analysis
- **20+ Technical Indicators**: RSI, MACD, Bollinger Bands, ATR, ADX, Stochastic, and more
- **Interactive Charts**: Candlestick charts with overlay indicators
- **Risk Metrics**: Sharpe Ratio, Sortino Ratio, Max Drawdown, Calmar Ratio

### Machine Learning Models
- **Random Forest Classifier**
- **XGBoost**
- **ARIMA + GARCH** (Time series + volatility)
- **LSTM Neural Network**
- **RNN (Recurrent Neural Network)**
- **Monte Carlo Simulation**
- **Ensemble Recommendations** via majority voting

### News & Sentiment Analysis
- **Multi-Source News**: Finviz and Google News integration
- **FinBERT Sentiment Analysis**: Financial domain-specific NLP model
- **Real-time News Aggregation**: Latest headlines with sentiment scores

### Strategy Backtesting
- **Rule-Based Signals**: Customizable technical indicator weights
- **Risk Management**: Stop-loss and take-profit levels
- **Performance Metrics**: Win rate, alpha, equity curve visualization

### Monte Carlo Simulations
- **Price Target Projections**: Probability of reaching specific price targets
- **Multi-Timeframe Analysis**: 3 days to 6 months projections
- **Advanced Statistics**: Student's t-distribution with volatility clustering

## 🏗️ Modular Architecture

```
stock_dashboard_modular/
├── app.py                      # Main Streamlit UI (500 lines)
├── modules/
│   ├── __init__.py            # Package initialization
│   ├── data_fetcher.py        # API calls & data retrieval (93 lines)
│   ├── indicators.py          # Technical indicators (89 lines)
│   ├── news_scraper.py        # Multi-source news (172 lines)
│   ├── sentiment.py           # FinBERT analysis (123 lines)
│   ├── strategy.py            # Trading signals (110 lines)
│   ├── risk_metrics.py        # Risk calculations (38 lines)
│   ├── backtester.py          # Strategy backtesting (99 lines)
│   ├── ml_models.py           # 6 ML algorithms (433 lines)
│   └── monte_carlo.py         # Simulations (123 lines)
├── .streamlit/
│   └── config.toml            # Streamlit configuration
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/stock-dashboard.git
cd stock-dashboard
```

2. **Install dependencies**
```bash
pip install -r requirements.txt --break-system-packages
```

3. **Run the dashboard**
```bash
streamlit run app.py
```

4. **Open in browser**
The dashboard will automatically open at `http://localhost:8501`

## 📦 Module Descriptions

### `data_fetcher.py`
- Fetches stock data from Yahoo Finance
- Retrieves Fear & Greed Index from CNN
- Implements caching for optimal performance

### `indicators.py`
- Calculates RSI, MACD, Bollinger Bands, ATR, ADX, Stochastic
- Validates data quality
- Handles missing data gracefully

### `news_scraper.py`
- Scrapes Finviz for ticker-specific news
- Fetches Google News via RSS
- Unified interface for multiple sources

### `sentiment.py`
- Loads and caches FinBERT model
- Analyzes financial news sentiment
- Returns confidence scores and labels

### `strategy.py`
- Generates BUY/SELL/HOLD signals
- Customizable indicator weights
- Trend strength modifiers

### `risk_metrics.py`
- Calculates Sharpe and Sortino ratios
- Maximum drawdown analysis
- Value at Risk (VaR) calculations

### `backtester.py`
- Simulates trading strategy
- Implements stop-loss and take-profit
- Tracks equity curve

### `ml_models.py`
- Trains 6 different ML models
- Feature engineering from technical indicators
- Ensemble voting system

### `monte_carlo.py`
- Price target probability simulations
- Multi-timeframe projections
- Fat-tailed distribution modeling

## ⚙️ Configuration

### Sidebar Options

**Market Cap Selection**
- Big Cap (>$100B): AAPL, MSFT, GOOGL, etc.
- Medium Cap ($10B-$100B): AMD, ADBE, PYPL, etc.
- Small Cap (<$10B): SOFI, HOOD, RKT, etc.

**Technical Indicators**
- RSI Period: 10-30 (default: 14)
- RSI Oversold/Overbought: 10-40 / 60-85
- SMA Short/Long: 10-40 / 30-200

**Signal Weights**
- Customize importance of each indicator
- Range: 0.0 to 5.0
- Default: RSI=2.0, MACD=1.5, SMA=1.0, BB=1.0

**Monte Carlo Settings**
- Simulation count: 500 to 10,000
- Max horizon: 90 to 730 days

**Backtest Settings**
- Initial capital: $1,000 to $1,000,000
- Confidence threshold: 0% to 50%
- Stop loss: 1% to 20%
- Take profit: 5% to 50%

## 📊 Usage Examples

### Basic Analysis
1. Select a ticker from the dropdown
2. View key metrics and price chart
3. Check rule-based trading signals

### ML Analysis
1. Click "🚀 Run ML Analysis"
2. Wait for 6 models to train (~1-2 minutes)
3. View ensemble recommendation

### News Sentiment
1. Select news source (Finviz or Google News)
2. Enable FinBERT for accurate sentiment
3. View sentiment breakdown

### Strategy Backtesting
1. Adjust indicator weights in sidebar
2. Set risk management parameters
3. View equity curve and trade history

### Monte Carlo Simulations
1. Set simulation count and horizon
2. View probability of reaching price targets
3. Analyze multi-timeframe projections

## 🔧 Dependencies

### Core Libraries
- `streamlit` - Web application framework
- `pandas` - Data manipulation
- `numpy` - Numerical computations
- `yfinance` - Stock data API
- `plotly` - Interactive charts

### ML & Statistics
- `scikit-learn` - Random Forest, preprocessing
- `xgboost` - Gradient boosting
- `tensorflow` - LSTM, RNN neural networks
- `statsmodels` - ARIMA time series
- `arch` - GARCH volatility modeling
- `scipy` - Statistical functions

### News & Sentiment
- `beautifulsoup4` - Web scraping
- `requests` - HTTP requests
- `feedparser` - RSS feed parsing
- `transformers` - FinBERT model
- `torch` - PyTorch backend

## ⚠️ Important Disclaimers

### NOT FINANCIAL ADVICE
This tool is for **educational purposes only**. It should not be used as the sole basis for investment decisions.

### Key Limitations
- Past performance does not guarantee future results
- ML models can overfit to historical patterns
- Market conditions change constantly
- Sentiment analysis is headline-based only
- News sources may have biases or delays

### Recommendations
1. **Paper trade first**: Test strategies for 3+ months before using real capital
2. **Diversify**: Don't rely on a single indicator or model
3. **Stay informed**: Follow company news and earnings
4. **Manage risk**: Always use stop-losses
5. **Consult experts**: Speak with licensed financial advisors

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

MIT License - See LICENSE file for details

## 👨‍💻 Author

**Sadiq**
- Enhanced Stock Dashboard v2.3 - Modular Edition

## 🙏 Acknowledgments

- Yahoo Finance for stock data
- Finviz for financial news
- Google News for news aggregation
- ProsusAI for FinBERT model
- Streamlit community for excellent framework

## 📧 Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check documentation in the `/docs` folder
- Review the code comments in each module

## 🔄 Version History

### v2.3 - Modular Edition (Current)
- Refactored into 9 specialized modules
- Improved performance and caching
- Better maintainability
- Enhanced documentation

### v2.2
- Added LSTM and RNN models
- Multi-source news integration
- FinBERT sentiment analysis

### v2.1
- Strategy backtesting
- Risk metrics dashboard
- Monte Carlo simulations

### v2.0
- Machine learning models
- Advanced technical indicators
- Interactive charts

### v1.0
- Basic technical analysis
- Rule-based signals
- Single stock analysis

---

**Made with ❤️ and Python | Built with Streamlit**
