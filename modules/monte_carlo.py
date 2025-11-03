"""
Monte Carlo Simulation Module
Advanced Monte Carlo simulations for price targets and projections
"""
import pandas as pd
import numpy as np
from scipy import stats


def estimate_days_to_target_advanced(df: pd.DataFrame, current_price: float,
                                     target_return: float, sims: int = 5000,
                                     max_days: int = 365):
    """Advanced Monte Carlo simulation"""
    returns = df['Close'].pct_change().dropna().values
    if len(returns) < 30:
        return {'probability': 0.0, 'median_days': None, '90pct_days': None, '10pct_days': None}
    
    weights = np.exp(np.linspace(-2, 0, len(returns)))
    weights /= weights.sum()
    mu = np.average(returns, weights=weights)
    sigma = np.sqrt(np.average((returns - mu)**2, weights=weights))
    
    if sigma == 0:
        return {'probability': 0.0, 'median_days': None, '90pct_days': None, '10pct_days': None}
    
    t_samples = stats.t.rvs(df=5, loc=mu, scale=sigma, size=(sims, max_days))
    vol_factor = np.ones((sims, max_days))
    for d in range(1, max_days):
        vol_factor[:, d] = 0.85 * vol_factor[:, d-1] + 0.15 * (1 + np.abs(t_samples[:, d-1]))
    t_samples *= vol_factor
    
    price_paths = current_price * np.cumprod(1 + t_samples, axis=1)
    threshold = current_price * (1 + target_return)
    hits = price_paths >= threshold
    first_hit = np.argmax(hits, axis=1) + 1
    no_hit_mask = ~hits.any(axis=1)
    first_hit = first_hit.astype(float)
    first_hit[no_hit_mask] = np.nan
    
    valid = ~np.isnan(first_hit)
    prob_reach = valid.mean()
    median_days = float(np.nanmedian(first_hit)) if prob_reach > 0 else None
    pct90 = float(np.nanpercentile(first_hit[valid], 90)) if prob_reach > 0 else None
    pct10 = float(np.nanpercentile(first_hit[valid], 10)) if prob_reach > 0 else None
    mean_days = float(np.nanmean(first_hit)) if prob_reach > 0 else None
    
    return {'probability': prob_reach, 'median_days': median_days, '90pct_days': pct90, '10pct_days': pct10, 'mean_days': mean_days}


def monte_carlo_price_simulation(df: pd.DataFrame, current_price: float, sims: int = 5000):
    """
    Monte Carlo simulation for price ranges across different timeframes
    
    Args:
        df: Historical price dataframe
        current_price: Current stock price
        sims: Number of simulations to run
    
    Returns:
        DataFrame with columns: Timeframe, Lowest Price, Median Price, Highest Price
    """
    returns = df['Close'].pct_change().dropna().values
    
    if len(returns) < 30:
        return pd.DataFrame()
    
    # Calculate weighted statistics for more recent data
    weights = np.exp(np.linspace(-2, 0, len(returns)))
    weights /= weights.sum()
    mu = np.average(returns, weights=weights)
    sigma = np.sqrt(np.average((returns - mu)**2, weights=weights))
    
    if sigma == 0:
        timeframes_data = []
        for name, days in [("3 Days", 3), ("1 Week", 7), ("2 Weeks", 14), 
                          ("1 Month", 30), ("3 Months", 90), ("6 Months", 180)]:
            timeframes_data.append({
                "Timeframe": name,
                "Lowest Price": f"${current_price:.2f}",
                "Median Price": f"${current_price:.2f}",
                "Highest Price": f"${current_price:.2f}"
            })
        return pd.DataFrame(timeframes_data)
    
    # Define timeframes
    timeframes = [
        ("3 Days", 3),
        ("1 Week", 7),
        ("2 Weeks", 14),
        ("1 Month", 30),
        ("3 Months", 90),
        ("6 Months", 180)
    ]
    
    simulation_results = []
    
    for timeframe_name, days in timeframes:
        # Generate simulations using Student's t-distribution
        t_samples = stats.t.rvs(df=5, loc=mu, scale=sigma, size=(sims, days))
        
        # Add volatility clustering
        vol_factor = np.ones((sims, days))
        for d in range(1, days):
            vol_factor[:, d] = 0.85 * vol_factor[:, d-1] + 0.15 * (1 + np.abs(t_samples[:, d-1]))
        t_samples *= vol_factor
        
        # Calculate final prices
        price_paths = current_price * np.cumprod(1 + t_samples, axis=1)
        final_prices = price_paths[:, -1]
        
        # Calculate percentiles
        lowest_price = np.percentile(final_prices, 5)   # Worst 5%
        median_price = np.percentile(final_prices, 50)  # Median
        highest_price = np.percentile(final_prices, 95) # Best 5%
        
        simulation_results.append({
            "Timeframe": timeframe_name,
            "Lowest Price": f"${lowest_price:.2f}",
            "Median Price": f"${median_price:.2f}",
            "Highest Price": f"${highest_price:.2f}"
        })
    
    return pd.DataFrame(simulation_results)
