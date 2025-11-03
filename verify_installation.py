#!/usr/bin/env python3
"""
Installation Verification Script
Tests that all modules can be imported and basic functionality works
"""

def test_imports():
    """Test all module imports"""
    print("🔍 Testing module imports...")
    
    try:
        from modules.data_fetcher import get_data_optimized
        print("  ✅ data_fetcher imported")
    except Exception as e:
        print(f"  ❌ data_fetcher failed: {e}")
        return False
    
    try:
        from modules.indicators import calc_indicators
        print("  ✅ indicators imported")
    except Exception as e:
        print(f"  ❌ indicators failed: {e}")
        return False
    
    try:
        from modules.strategy import rule_based_signal_v2
        print("  ✅ strategy imported")
    except Exception as e:
        print(f"  ❌ strategy failed: {e}")
        return False
    
    try:
        from modules.risk_metrics import calculate_risk_metrics
        print("  ✅ risk_metrics imported")
    except Exception as e:
        print(f"  ❌ risk_metrics failed: {e}")
        return False
    
    print("\n✅ All module imports successful!\n")
    return True


def test_data_fetch():
    """Test data fetching"""
    print("📊 Testing data fetch...")
    
    try:
        from modules.data_fetcher import get_data_optimized
        hist, info = get_data_optimized("AAPL", period="1mo", interval="1d")
        
        if hist.empty:
            print("  ❌ Data fetch returned empty DataFrame")
            return False
        
        print(f"  ✅ Fetched {len(hist)} data points for AAPL")
        return True
    except Exception as e:
        print(f"  ❌ Data fetch failed: {e}")
        return False


def test_indicators():
    """Test indicator calculations"""
    print("\n📈 Testing indicator calculations...")
    
    try:
        from modules.data_fetcher import get_data_optimized
        from modules.indicators import calc_indicators
        
        hist, _ = get_data_optimized("AAPL", period="1mo", interval="1d")
        df = calc_indicators(hist)
        
        required = ['RSI', 'MACD', 'SMA_short', 'SMA_long', 'BB_upper', 'BB_lower']
        for indicator in required:
            if indicator not in df.columns:
                print(f"  ❌ Missing indicator: {indicator}")
                return False
        
        print(f"  ✅ All {len(required)} indicators calculated")
        return True
    except Exception as e:
        print(f"  ❌ Indicator calculation failed: {e}")
        return False


def test_strategy():
    """Test strategy signal generation"""
    print("\n🎯 Testing strategy signals...")
    
    try:
        from modules.data_fetcher import get_data_optimized
        from modules.indicators import calc_indicators
        from modules.strategy import rule_based_signal_v2
        
        hist, _ = get_data_optimized("AAPL", period="1mo", interval="1d")
        df = calc_indicators(hist)
        
        recommendation, signals, confidence, scores = rule_based_signal_v2(df)
        
        if recommendation not in ["BUY", "SELL", "HOLD", "STRONG BUY", "STRONG SELL"]:
            print(f"  ❌ Invalid recommendation: {recommendation}")
            return False
        
        print(f"  ✅ Generated signal: {recommendation} (confidence: {confidence:.1f}%)")
        return True
    except Exception as e:
        print(f"  ❌ Strategy signal failed: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 50)
    print("INSTALLATION VERIFICATION")
    print("=" * 50)
    print()
    
    results = []
    
    # Run tests
    results.append(("Module Imports", test_imports()))
    results.append(("Data Fetch", test_data_fetch()))
    results.append(("Indicators", test_indicators()))
    results.append(("Strategy", test_strategy()))
    
    # Print summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")
    
    print()
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Installation is working correctly.")
        print("\nNext steps:")
        print("  1. Run: streamlit run app.py")
        print("  2. Open: http://localhost:8501")
        print("  3. Enjoy your modular stock dashboard!")
    else:
        print("\n⚠️  Some tests failed. Please check:")
        print("  1. All dependencies installed: pip install -r requirements.txt")
        print("  2. Internet connection available (for data fetch)")
        print("  3. Error messages above for details")
    
    return passed == total


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
