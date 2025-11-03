"""
Risk Metrics Module
Calculate comprehensive risk metrics including Sharpe, Sortino, Max Drawdown, Calmar
"""
import pandas as pd
import numpy as np


def calculate_risk_metrics(df: pd.DataFrame, risk_free_rate: float = 0.04):
    """Calculate comprehensive risk metrics"""
    returns = df['Close'].pct_change().dropna()
    if len(returns) < 2:
        return None
    
    annual_return = (df['Close'].iloc[-1] / df['Close'].iloc[0]) ** (252/len(df)) - 1
    volatility = returns.std() * np.sqrt(252)
    excess_returns = returns - risk_free_rate/252
    sharpe = np.sqrt(252) * excess_returns.mean() / returns.std() if returns.std() > 0 else 0
    downside = returns[returns < 0]
    sortino = np.sqrt(252) * excess_returns.mean() / downside.std() if len(downside) > 0 and downside.std() > 0 else 0
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()
    calmar = annual_return / abs(max_dd) if max_dd != 0 else 0
    var_95 = returns.quantile(0.05)
    cvar_95 = returns[returns <= var_95].mean()
    
    return {
        'annual_return': annual_return,
        'volatility': volatility,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_drawdown': max_dd,
        'calmar': calmar,
        'var_95': var_95,
        'cvar_95': cvar_95
    }
