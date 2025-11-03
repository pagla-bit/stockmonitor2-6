"""
Technical Indicators Module
Calculates all technical indicators: RSI, MACD, Bollinger Bands, ATR, ADX, Stochastic
"""
import pandas as pd
import numpy as np


def calc_indicators(df: pd.DataFrame,
                    rsi_period=14,
                    macd_fast=12, macd_slow=26, macd_signal=9,
                    sma_short=20, sma_long=50,
                    bb_period=20, atr_period=14, adx_period=14):
    """Calculate technical indicators with validation"""
    df = df.copy()
    
    # Moving Averages
    df["SMA_short"] = df["Close"].rolling(sma_short).mean()
    df["SMA_long"] = df["Close"].rolling(sma_long).mean()
    df["EMA_short"] = df["Close"].ewm(span=sma_short, adjust=False).mean()
    
    # RSI
    delta = df["Close"].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(com=rsi_period - 1, adjust=False).mean()
    ma_down = down.ewm(com=rsi_period - 1, adjust=False).mean()
    rs = ma_up / ma_down
    df["RSI"] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_fast = df["Close"].ewm(span=macd_fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=macd_slow, adjust=False).mean()
    df["MACD"] = ema_fast - ema_slow
    df["MACD_signal"] = df["MACD"].ewm(span=macd_signal, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]
    
    # Bollinger Bands
    bb_mid = df["Close"].rolling(bb_period).mean()
    bb_std = df["Close"].rolling(bb_period).std()
    df["BB_upper"] = bb_mid + 2 * bb_std
    df["BB_lower"] = bb_mid - 2 * bb_std
    df["BB_mid"] = bb_mid
    df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / bb_mid
    df["BB_position"] = (df["Close"] - df["BB_lower"]) / (df["BB_upper"] - df["BB_lower"])
    
    # ATR
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    df["ATR"] = tr.ewm(span=atr_period, adjust=False).mean()
    df["ATR_pct"] = (df["ATR"] / df["Close"]) * 100
    
    # ADX
    plus_dm = df["High"].diff()
    minus_dm = -df["Low"].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    tr_smooth = tr.ewm(span=adx_period).mean()
    plus_di = 100 * (plus_dm.ewm(span=adx_period).mean() / tr_smooth)
    minus_di = 100 * (minus_dm.ewm(span=adx_period).mean() / tr_smooth)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    df["ADX"] = dx.ewm(span=adx_period).mean()
    df["DI_plus"] = plus_di
    df["DI_minus"] = minus_di
    
    # Stochastic Oscillator
    low_14 = df["Low"].rolling(14).min()
    high_14 = df["High"].rolling(14).max()
    df["Stoch_K"] = 100 * ((df["Close"] - low_14) / (high_14 - low_14))
    df["Stoch_D"] = df["Stoch_K"].rolling(3).mean()
    
    return df


def validate_indicators(df: pd.DataFrame, min_valid_pct=0.7):
    """Validate that indicators have sufficient valid data"""
    required = ['RSI', 'MACD', 'ATR', 'ADX']
    validation_results = {}
    
    for col in required:
        if col in df:
            valid_pct = df[col].notna().mean()
            validation_results[col] = valid_pct
            if valid_pct < min_valid_pct:
                return False, f"{col} has only {valid_pct*100:.1f}% valid data (need {min_valid_pct*100}%)"
    
    return True, validation_results
