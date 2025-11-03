# 🚀 Quick Start Guide

Get the Enhanced Stock Dashboard running in 5 minutes!

## Step 1: Prerequisites

Make sure you have Python 3.8+ installed:
```bash
python --version
```

## Step 2: Clone/Download

```bash
git clone https://github.com/yourusername/stock-dashboard.git
cd stock-dashboard
```

Or download and extract the ZIP file.

## Step 3: Install Dependencies

### Option A: Standard Installation
```bash
pip install -r requirements.txt
```

### Option B: With Break System Packages (Linux/Mac)
```bash
pip install -r requirements.txt --break-system-packages
```

### Option C: Using Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Step 4: Run the Dashboard

```bash
streamlit run app.py
```

The dashboard will open automatically at: http://localhost:8501

## Step 5: First Analysis

1. **Select a stock**: Choose from dropdown or add your own ticker
2. **View analysis**: See charts, indicators, and metrics
3. **Check news**: Select Finviz or Google News
4. **Run ML**: Click "🚀 Run ML Analysis" button
5. **View signals**: Check BUY/SELL/HOLD recommendations

## 🔧 Quick Configuration

### Change Tickers
Edit in sidebar → "Tickers (comma separated)"
```
AAPL, MSFT, TSLA, NVDA
```

### Adjust Indicators
Sidebar → "📊 Indicator Parameters"
- RSI Period: 14 (default)
- SMA Short: 20 (default)
- SMA Long: 50 (default)

### Enable FinBERT
Sidebar → "🤖 Sentiment Analysis"
- Check "Use FinBERT" for accurate sentiment
- Note: Takes 10-15 seconds per analysis

## 🐛 Common Issues

### Issue: Module Not Found
**Solution**: Install missing module
```bash
pip install <module_name> --break-system-packages
```

### Issue: TensorFlow Error
**Solution**: Install compatible version
```bash
pip install tensorflow==2.14.0
```

### Issue: FinBERT Not Loading
**Solution**: Install transformers and torch
```bash
pip install transformers torch --break-system-packages
```

### Issue: Port Already in Use
**Solution**: Use different port
```bash
streamlit run app.py --server.port 8502
```

### Issue: Slow Performance
**Solution**: 
1. Reduce simulation count in sidebar
2. Use shorter lookback period (6mo instead of 5y)
3. Disable FinBERT if not needed

## 📊 Testing the Dashboard

### Test 1: Basic Functionality
1. Load AAPL
2. View price chart
3. Check technical indicators

### Test 2: News & Sentiment
1. Select "Finviz" source
2. Enable FinBERT
3. View sentiment scores

### Test 3: ML Analysis
1. Click "Run ML Analysis"
2. Wait 1-2 minutes
3. View ensemble recommendation

### Test 4: Backtesting
1. Adjust indicator weights
2. Set risk parameters
3. View equity curve

## 🎯 Next Steps

- Read the full [README.md](README.md)
- Explore different tickers
- Customize indicator weights
- Test different strategies
- Paper trade for 3+ months before using real capital

## 💡 Tips

1. **Start with big caps**: AAPL, MSFT, GOOGL are well-covered
2. **Use multiple indicators**: Don't rely on just one signal
3. **Check sentiment**: News can drive short-term movements
4. **Compare ML models**: Look for consensus across models
5. **Monitor risk metrics**: Sharpe ratio, max drawdown, etc.

## 📞 Need Help?

- Check [README.md](README.md) for detailed docs
- Review code comments in each module
- Open an issue on GitHub
- Read error messages carefully

## ⚠️ Remember

**This is NOT financial advice!** 
- Always do your own research
- Consult with financial advisors
- Use paper trading first
- Never invest more than you can afford to lose

---

**Happy Trading! 📈**
