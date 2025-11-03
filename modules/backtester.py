"""
Backtesting Module
Strategy backtesting with risk management (stop loss, take profit)
"""
import pandas as pd
from .strategy import rule_based_signal_v2


def backtest_strategy(df: pd.DataFrame, weights: dict, initial_capital: float = 10000,
                     confidence_threshold: float = 20, stop_loss_pct: float = 0.05,
                     take_profit_pct: float = 0.15):
    """Backtest strategy with risk management"""
    df = df.copy()
    positions = []
    capital = initial_capital
    shares = 0
    entry_price = 0
    transaction_cost = 0.001
    buy_signals = []
    sell_signals = []
    equity_curve = []
    
    for i in range(50, len(df)):
        window = df.iloc[:i+1]
        if len(window) < 50:
            continue
        
        rec, _, conf, scores = rule_based_signal_v2(window, weights=weights)
        current_price = df['Close'].iloc[i]
        current_date = df.index[i]
        
        portfolio_value = shares * current_price if shares > 0 else capital
        equity_curve.append({'date': current_date, 'value': portfolio_value})
        
        if shares > 0:
            return_pct = (current_price - entry_price) / entry_price
            if return_pct <= -stop_loss_pct:
                capital = shares * current_price * (1 - transaction_cost)
                positions.append(('STOP_LOSS', current_date, current_price, capital, return_pct))
                sell_signals.append(i)
                shares = 0
                entry_price = 0
                continue
            if return_pct >= take_profit_pct:
                capital = shares * current_price * (1 - transaction_cost)
                positions.append(('TAKE_PROFIT', current_date, current_price, capital, return_pct))
                sell_signals.append(i)
                shares = 0
                entry_price = 0
                continue
        
        if rec in ['BUY', 'STRONG BUY'] and shares == 0 and abs(conf) > confidence_threshold:
            shares = (capital * (1 - transaction_cost)) / current_price
            entry_price = current_price
            capital = 0
            positions.append(('BUY', current_date, current_price, shares, conf))
            buy_signals.append(i)
        elif rec in ['SELL', 'STRONG SELL'] and shares > 0:
            return_pct = (current_price - entry_price) / entry_price
            capital = shares * current_price * (1 - transaction_cost)
            positions.append(('SELL', current_date, current_price, capital, return_pct))
            sell_signals.append(i)
            shares = 0
            entry_price = 0
    
    if shares > 0:
        final_price = df['Close'].iloc[-1]
        return_pct = (final_price - entry_price) / entry_price
        capital = shares * final_price * (1 - transaction_cost)
        positions.append(('FINAL_CLOSE', df.index[-1], final_price, capital, return_pct))
    
    final_capital = capital if shares == 0 else shares * df['Close'].iloc[-1]
    total_return = (final_capital - initial_capital) / initial_capital
    buy_hold_shares = (initial_capital * (1 - transaction_cost)) / df['Close'].iloc[50]
    buy_hold_final = buy_hold_shares * df['Close'].iloc[-1] * (1 - transaction_cost)
    buy_hold_return = (buy_hold_final - initial_capital) / initial_capital
    
    trades_with_returns = [p for p in positions if len(p) > 4 and isinstance(p[4], (int, float))]
    winning_trades = [p for p in trades_with_returns if p[4] > 0]
    win_rate = len(winning_trades) / len(trades_with_returns) if trades_with_returns else 0
    wins = [p[4] for p in winning_trades]
    losses = [p[4] for p in trades_with_returns if p[4] <= 0]
    avg_win = sum(wins) / len(wins) if wins else 0
    avg_loss = sum(losses) / len(losses) if losses else 0
    
    return {
        'final_capital': final_capital,
        'total_return': total_return,
        'buy_hold_return': buy_hold_return,
        'alpha': total_return - buy_hold_return,
        'num_trades': len(positions),
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'positions': positions,
        'buy_signals': buy_signals,
        'sell_signals': sell_signals,
        'equity_curve': equity_curve
    }
