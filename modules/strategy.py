"""
Trading Strategy Module
Rule-based trading signal generation with technical indicators
"""
import pandas as pd
import numpy as np


def rule_based_signal_v2(df: pd.DataFrame,
                         rsi_oversold=30,
                         rsi_overbought=70,
                         weights=None):
    """Enhanced signal generation with proper modifiers"""
    if weights is None:
        weights = {'RSI': 2.0, 'MACD': 1.5, 'SMA': 1.0, 'BB': 1.0, 'Stoch': 0.8, 'Volume': 0.5, 'ADX': 1.0}
    
    if len(df) < 3:
        return "HOLD", [], 0.0, {'buy': 0, 'sell': 0}
    
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    buy_score = 0.0
    sell_score = 0.0
    trend_multiplier = 1.0
    volume_multiplier = 1.0
    signals = []
    
    # RSI
    if not np.isnan(latest['RSI']):
        if latest['RSI'] < rsi_oversold:
            buy_score += weights['RSI']
            signals.append(("RSI oversold", "BUY", weights['RSI'], f"RSI={latest['RSI']:.1f}"))
        elif latest['RSI'] > rsi_overbought:
            sell_score += weights['RSI']
            signals.append(("RSI overbought", "SELL", weights['RSI'], f"RSI={latest['RSI']:.1f}"))
    
    # MACD
    if not np.isnan(latest['MACD']) and not np.isnan(prev['MACD']):
        if (prev['MACD'] < prev['MACD_signal']) and (latest['MACD'] > latest['MACD_signal']):
            buy_score += weights['MACD']
            signals.append(("MACD bullish crossover", "BUY", weights['MACD'], ""))
        elif (prev['MACD'] > prev['MACD_signal']) and (latest['MACD'] < latest['MACD_signal']):
            sell_score += weights['MACD']
            signals.append(("MACD bearish crossover", "SELL", weights['MACD'], ""))
    
    # SMA
    if not np.isnan(latest['SMA_long']):
        if latest['Close'] > latest['SMA_long']:
            buy_score += weights['SMA']
            signals.append(("Price above long SMA", "BUY", weights['SMA'], ""))
        else:
            sell_score += weights['SMA']
            signals.append(("Price below long SMA", "SELL", weights['SMA'], ""))
    
    # Bollinger Bands
    if not np.isnan(latest['BB_lower']) and not np.isnan(latest['BB_upper']):
        if latest['Close'] < latest['BB_lower']:
            buy_score += weights['BB']
            signals.append(("Price below BB lower", "BUY", weights['BB'], ""))
        elif latest['Close'] > latest['BB_upper']:
            sell_score += weights['BB']
            signals.append(("Price above BB upper", "SELL", weights['BB'], ""))
    
    # Stochastic
    if 'Stoch_K' in latest and not np.isnan(latest['Stoch_K']):
        if latest['Stoch_K'] < 20:
            buy_score += weights.get('Stoch', 0.8)
            signals.append(("Stochastic oversold", "BUY", weights.get('Stoch', 0.8), ""))
        elif latest['Stoch_K'] > 80:
            sell_score += weights.get('Stoch', 0.8)
            signals.append(("Stochastic overbought", "SELL", weights.get('Stoch', 0.8), ""))
    
    # Volume
    vol_avg20 = df['Volume'].rolling(20).mean().iloc[-1] if len(df) >= 20 else df['Volume'].mean()
    if vol_avg20 > 0 and latest['Volume'] > vol_avg20 * 1.5:
        volume_multiplier = 1.3
        signals.append(("High volume", "CONFIRM", 0, ""))
    
    # ADX
    if not np.isnan(latest['ADX']):
        if latest['ADX'] > 25:
            trend_multiplier = 1.0 + min((latest['ADX'] - 25) / 100, 0.5)
            signals.append(("Strong trend", "AMPLIFY", 0, f"ADX={latest['ADX']:.1f}"))
        elif latest['ADX'] < 20:
            trend_multiplier = 0.6
            signals.append(("Weak trend", "DAMPEN", 0, f"ADX={latest['ADX']:.1f}"))
    
    buy_score = buy_score * trend_multiplier * volume_multiplier
    sell_score = sell_score * trend_multiplier * volume_multiplier
    
    total_score = buy_score + sell_score
    confidence = ((buy_score - sell_score) / total_score * 100) if total_score > 0 else 0.0
    
    net_score = buy_score - sell_score
    strong_threshold = max(weights.values()) * 2.5
    
    if net_score > strong_threshold:
        recommendation = "STRONG BUY"
    elif net_score > 0.5:
        recommendation = "BUY"
    elif net_score < -strong_threshold:
        recommendation = "STRONG SELL"
    elif net_score < -0.5:
        recommendation = "SELL"
    else:
        recommendation = "HOLD"
    
    raw_scores = {'buy': buy_score, 'sell': sell_score, 'net': net_score}
    return recommendation, signals, confidence, raw_scores
