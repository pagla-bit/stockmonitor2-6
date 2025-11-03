# 🏗️ Architecture Documentation

## Overview

The Enhanced Stock Dashboard follows a **modular monorepo** architecture, separating concerns into specialized modules while maintaining a single, cohesive application.

## Design Principles

### 1. Separation of Concerns
Each module handles ONE specific responsibility:
- Data fetching ≠ Analysis
- News scraping ≠ Sentiment analysis
- Strategy logic ≠ UI rendering

### 2. Single Source of Truth
- Each function exists in exactly ONE module
- No duplicate logic across files
- Clear import paths

### 3. Composability
- Modules can be used independently
- Easy to test individual components
- Simple to extend or replace modules

### 4. Performance Optimization
- Streamlit caching at module level
- Lazy loading of heavy dependencies
- Efficient data passing between modules

## Module Dependency Graph

```
app.py (UI Layer)
    │
    ├─── data_fetcher.py ────┐
    │                        │
    ├─── indicators.py ──────┤
    │                        ├──> DataFrames & Metrics
    ├─── news_scraper.py ────┤
    │                        │
    ├─── sentiment.py ───────┘
    │
    ├─── strategy.py ────────> Trading Signals
    │
    ├─── risk_metrics.py ────> Risk Calculations
    │
    ├─── backtester.py ──────> Strategy Testing
    │         │
    │         └──> strategy.py
    │
    ├─── ml_models.py ───────> ML Predictions
    │         │
    │         └──> indicators.py
    │
    └─── monte_carlo.py ─────> Simulations
```

## Data Flow

### 1. Data Acquisition
```
User Input (Ticker) 
    → data_fetcher.get_data_optimized() 
    → Raw OHLCV DataFrame
```

### 2. Feature Engineering
```
Raw DataFrame 
    → indicators.calc_indicators() 
    → Enhanced DataFrame with 20+ indicators
```

### 3. Signal Generation
```
Enhanced DataFrame 
    → strategy.rule_based_signal_v2() 
    → BUY/SELL/HOLD signal
```

### 4. ML Analysis
```
Enhanced DataFrame 
    → ml_models.prepare_ml_features() 
    → Feature Matrix
    → ml_models.train_*() for each algorithm
    → ml_models.calculate_ensemble_recommendation()
    → Final ML recommendation
```

### 5. Risk Assessment
```
Enhanced DataFrame 
    → risk_metrics.calculate_risk_metrics() 
    → Sharpe, Sortino, Max DD, etc.
```

### 6. Backtesting
```
Enhanced DataFrame + Strategy Weights
    → backtester.backtest_strategy()
    → Equity curve + Trade history
```

### 7. News & Sentiment
```
Ticker 
    → news_scraper.get_news_from_source()
    → List of news articles
    → sentiment.analyze_news_sentiment_bert_only()
    → Articles with sentiment scores
```

## Module Details

### Core Modules (Foundation)

#### `data_fetcher.py`
**Purpose**: External API integration  
**Dependencies**: yfinance, requests  
**Caching**: 1 hour TTL  
**Key Functions**:
- `get_data_optimized()` - Fetch stock OHLCV data
- `get_spy_data()` - Market benchmark data
- `get_fear_greed_index()` - Market sentiment

#### `indicators.py`
**Purpose**: Technical analysis calculations  
**Dependencies**: pandas, numpy  
**Caching**: None (fast computation)  
**Key Functions**:
- `calc_indicators()` - Calculate 20+ indicators
- `validate_indicators()` - Data quality checks

### Analysis Modules (Business Logic)

#### `strategy.py`
**Purpose**: Trading signal generation  
**Dependencies**: indicators.py  
**Key Functions**:
- `rule_based_signal_v2()` - Weighted signal calculation
- Returns: Signal, confidence, breakdown

#### `risk_metrics.py`
**Purpose**: Portfolio risk calculations  
**Dependencies**: scipy  
**Key Functions**:
- `calculate_risk_metrics()` - 8 risk metrics
- Returns: Dict with all metrics

#### `backtester.py`
**Purpose**: Historical strategy testing  
**Dependencies**: strategy.py  
**Key Functions**:
- `backtest_strategy()` - Full backtest with risk management
- Returns: Performance stats + equity curve

### ML/AI Modules (Prediction)

#### `ml_models.py`
**Purpose**: Machine learning predictions  
**Dependencies**: scikit-learn, xgboost, tensorflow, statsmodels  
**Key Functions**:
- `prepare_ml_features()` - Feature engineering
- `train_*()` - Individual model training
- `run_ml_analysis()` - Train all 6 models
- `calculate_ensemble_recommendation()` - Voting system

#### `monte_carlo.py`
**Purpose**: Probabilistic forecasting  
**Dependencies**: scipy.stats  
**Key Functions**:
- `estimate_days_to_target_advanced()` - Price target simulation
- `monte_carlo_price_simulation()` - Multi-timeframe projections

### External Data Modules (Integration)

#### `news_scraper.py`
**Purpose**: Multi-source news aggregation  
**Dependencies**: beautifulsoup4, feedparser, requests  
**Caching**: 30 minutes TTL  
**Key Functions**:
- `scrape_finviz_news()` - Finviz scraper
- `scrape_google_news()` - Google News RSS parser
- `get_news_from_source()` - Unified interface

#### `sentiment.py`
**Purpose**: Financial sentiment analysis  
**Dependencies**: transformers, torch  
**Caching**: Model cached globally  
**Key Functions**:
- `load_finbert_model()` - Model initialization
- `get_finbert_sentiment()` - Single text analysis
- `analyze_news_sentiment_bert_only()` - Batch analysis

## Performance Considerations

### Caching Strategy

1. **Data Layer**: 1 hour TTL
   - Stock prices change slowly
   - Reduces API calls

2. **News Layer**: 30 minutes TTL
   - News updates frequently
   - Balance freshness vs. performance

3. **Model Layer**: Session-based
   - Models persist across reruns
   - Only reload on app restart

### Memory Management

- **Heavy imports**: Lazy loading (only when needed)
- **Large DataFrames**: Pass by reference, not copy
- **ML models**: Cache globally, train once

### Computation Optimization

- **Vectorized operations**: NumPy/Pandas throughout
- **Parallel processing**: ThreadPoolExecutor for news
- **Early returns**: Validate data before heavy computation

## Extension Points

### Adding a New Indicator

1. Add calculation to `indicators.py`
2. Update `validate_indicators()` if needed
3. Use in `strategy.py` or `ml_models.py`

Example:
```python
# In indicators.py
def calc_indicators(df, ...):
    # ... existing code ...
    df['My_Indicator'] = my_calculation(df['Close'])
    return df
```

### Adding a New ML Model

1. Create `train_my_model()` in `ml_models.py`
2. Add to `run_ml_analysis()` function
3. Model auto-included in ensemble

Example:
```python
# In ml_models.py
def train_my_model(X, y):
    model = MyModel()
    model.fit(X, y)
    return model, params_str, metrics_str, prediction, confidence
```

### Adding a New News Source

1. Create `scrape_my_source()` in `news_scraper.py`
2. Add to `get_news_from_source()` switch
3. Update UI dropdown in `app.py`

Example:
```python
# In news_scraper.py
@st.cache_data(ttl=1800)
def scrape_my_source(ticker, max_news=10):
    # Scraping logic
    return news_list
```

## Testing Strategy

### Unit Tests (Recommended)
```python
# test_indicators.py
from modules.indicators import calc_indicators

def test_rsi_calculation():
    df = pd.DataFrame({'Close': [100, 102, 101, 103, 102]})
    result = calc_indicators(df)
    assert 'RSI' in result.columns
    assert result['RSI'].notna().any()
```

### Integration Tests
```python
# test_integration.py
def test_full_pipeline():
    hist, info = get_data_optimized("AAPL")
    df = calc_indicators(hist)
    signal = rule_based_signal_v2(df)
    assert signal[0] in ["BUY", "SELL", "HOLD"]
```

## Error Handling

### Graceful Degradation
- If FinBERT fails → Show neutral sentiment
- If ML model fails → Skip that model
- If news scraping fails → Show empty table

### User Feedback
- Loading spinners for long operations
- Error messages with context
- Warnings for data quality issues

## Security Considerations

1. **No sensitive data**: All data is public
2. **API rate limits**: Cached to prevent abuse
3. **Input validation**: Ticker symbols sanitized
4. **XSS protection**: Streamlit handles automatically

## Future Improvements

### Planned Features
1. Database integration for historical signals
2. Real-time WebSocket price updates
3. Alert system for signal changes
4. Export functionality (CSV, PDF reports)
5. Multi-stock portfolio analysis

### Performance Enhancements
1. Async data fetching
2. GPU acceleration for ML models
3. Redis caching for distributed deployment
4. Lazy loading of chart components

### Code Quality
1. Type hints throughout
2. Comprehensive docstrings
3. Unit test coverage >80%
4. CI/CD pipeline

## Contributing Guidelines

### Before Submitting PR

1. Run tests: `pytest tests/`
2. Check style: `flake8 modules/`
3. Update docs if needed
4. Add example in QUICKSTART if new feature

### Code Style

- Follow PEP 8
- Use descriptive variable names
- Add docstrings to all functions
- Keep functions under 50 lines

### Commit Messages

```
feat: Add new RSI divergence indicator
fix: Correct MACD calculation bug
docs: Update architecture diagram
refactor: Extract ML feature engineering
```

---

**Version**: 2.3 Modular  
**Last Updated**: 2025-01-30  
**Maintained by**: Sadiq
