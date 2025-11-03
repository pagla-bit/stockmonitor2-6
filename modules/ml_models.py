"""
Machine Learning Models Module
Contains Random Forest, XGBoost, ARIMA+GARCH, LSTM, RNN, and Monte Carlo simulation
"""
import pandas as pd
import numpy as np
from scipy import stats
import streamlit as st

# ML libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, SimpleRNN
from tensorflow.keras.optimizers import Adam


def prepare_ml_features(df: pd.DataFrame, lookback=60):
    """
    Prepare features for ML models using technical indicators
    Returns: X (features), y (labels), feature_names
    """
    df = df.copy()
    
    # Calculate returns for labeling
    df['Returns_5d'] = df['Close'].pct_change(5).shift(-5)
    
    # Create labels: BUY (1), HOLD (0), SELL (-1)
    df['Label'] = 0
    df.loc[df['Returns_5d'] > 0.02, 'Label'] = 1  # BUY if >2% gain
    df.loc[df['Returns_5d'] < -0.02, 'Label'] = -1  # SELL if >2% loss
    
    # Features from indicators
    feature_cols = ['RSI', 'MACD', 'MACD_signal', 'MACD_hist', 
                    'SMA_short', 'SMA_long', 'BB_upper', 'BB_lower', 'BB_width',
                    'ATR', 'ATR_pct', 'ADX', 'Stoch_K', 'Stoch_D']
    
    # Add lagged features
    for col in ['Close', 'Volume']:
        for lag in [1, 5, 10]:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
            feature_cols.append(f'{col}_lag_{lag}')
    
    # Add momentum features
    df['Momentum_5'] = df['Close'].pct_change(5)
    df['Momentum_10'] = df['Close'].pct_change(10)
    feature_cols.extend(['Momentum_5', 'Momentum_10'])
    
    # Drop NaN rows
    df = df.dropna()
    
    if len(df) < lookback + 10:
        return None, None, None, None
    
    X = df[feature_cols].values
    y = df['Label'].values
    
    return X, y, feature_cols, df['Returns_5d'].values


def train_random_forest(X, y):
    """Train Random Forest Classifier"""
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X, y)
    
    # Predictions
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)
    
    # Metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
    
    # AUC for multiclass
    try:
        auc = roc_auc_score(y, y_pred_proba, multi_class='ovr', average='weighted')
    except:
        auc = 0.0
    
    params_str = "n_estimators=100, max_depth=10"
    metrics_str = f"Acc:{accuracy:.2%} Prec:{precision:.2%} Rec:{recall:.2%} F1:{f1:.2%} AUC:{auc:.2%}"
    
    return model, params_str, metrics_str, y_pred[-1], y_pred_proba[-1]


def train_xgboost(X, y):
    """Train XGBoost Classifier"""
    # Convert labels to 0, 1, 2 for XGBoost
    y_encoded = y + 1  # -1 -> 0, 0 -> 1, 1 -> 2
    
    model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, 
                              random_state=42, n_jobs=-1, verbosity=0)
    model.fit(X, y_encoded)
    
    y_pred_encoded = model.predict(X)
    y_pred = y_pred_encoded - 1  # Convert back
    y_pred_proba = model.predict_proba(X)
    
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
    
    try:
        auc = roc_auc_score(y, y_pred_proba, multi_class='ovr', average='weighted')
    except:
        auc = 0.0
    
    params_str = "learning_rate=0.1, n_estimators=100, max_depth=6"
    metrics_str = f"Acc:{accuracy:.2%} Prec:{precision:.2%} Rec:{recall:.2%} F1:{f1:.2%} AUC:{auc:.2%}"
    
    return model, params_str, metrics_str, y_pred[-1], y_pred_proba[-1]


def train_arima_garch(df: pd.DataFrame):
    """Train ARIMA for price and GARCH for volatility"""
    returns = df['Close'].pct_change().dropna()
    current_price = df['Close'].iloc[-1]
    
    # ARIMA for price prediction (using log prices for better stability)
    try:
        log_prices = np.log(df['Close'])
        arima_model = ARIMA(log_prices, order=(2, 1, 2))
        arima_fit = arima_model.fit()
        log_forecast = arima_fit.forecast(steps=5)
        predicted_price = np.exp(log_forecast.iloc[-1])
        predicted_return = (predicted_price - current_price) / current_price
    except:
        predicted_return = returns.tail(10).mean() * 5
    
    # GARCH for volatility
    try:
        garch_model = arch_model(returns.dropna() * 100, vol='Garch', p=1, q=1)
        garch_fit = garch_model.fit(disp='off')
        garch_forecast = garch_fit.forecast(horizon=5)
        predicted_volatility = np.sqrt(garch_forecast.variance.values[-1, :].mean()) / 100
    except:
        predicted_volatility = returns.std() * np.sqrt(5)
    
    # Risk-adjusted thresholds based on volatility
    buy_threshold = max(0.005, predicted_volatility * 0.5)
    sell_threshold = -buy_threshold
    
    # Calculate confidence based on signal strength relative to volatility
    if predicted_volatility > 0:
        signal_strength = abs(predicted_return) / predicted_volatility
        base_confidence = min(0.9, 0.5 + signal_strength * 0.2)
    else:
        base_confidence = 0.5
    
    # Classification with dynamic thresholds
    if predicted_return > buy_threshold:
        prediction = 1  # BUY
        confidence = base_confidence
    elif predicted_return < sell_threshold:
        prediction = -1  # SELL
        confidence = base_confidence
    else:
        prediction = 0  # HOLD
        confidence = 0.5
    
    # Add directional bias based on recent trend
    recent_trend = (df['Close'].iloc[-1] - df['Close'].iloc[-5]) / df['Close'].iloc[-5]
    if abs(recent_trend) > 0.02:
        if (recent_trend > 0 and prediction >= 0) or (recent_trend < 0 and prediction <= 0):
            confidence = min(0.95, confidence + 0.1)
    
    params_str = "ARIMA(2,1,2) + GARCH(1,1), horizon=5d, adaptive thresholds"
    metrics_str = f"Pred:{predicted_return:.2%} Vol:{predicted_volatility:.2%} Threshold:±{buy_threshold:.2%}"
    
    return None, params_str, metrics_str, prediction, confidence


def train_lstm(X, y, seq_length=60):
    """Train LSTM model"""
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create sequences - need to align y properly
    X_seq_list = []
    y_seq_list = []
    
    for i in range(len(X_scaled) - seq_length):
        X_seq_list.append(X_scaled[i:i+seq_length])
        y_seq_list.append(y[i+seq_length])  # Target is the label AFTER the sequence
    
    X_seq = np.array(X_seq_list)
    y_seq = np.array(y_seq_list)
    
    if len(X_seq) < 10:
        return None, "Insufficient data", "N/A", 0, 0.5
    
    # Encode labels: -1 -> 0, 0 -> 1, 1 -> 2
    y_seq_encoded = y_seq + 1
    
    # Build model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(seq_length, X.shape[1])),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(3, activation='softmax')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Train
    model.fit(X_seq, y_seq_encoded, epochs=20, batch_size=32, verbose=0)
    
    # Predict
    y_pred_proba = model.predict(X_seq, verbose=0)
    y_pred_encoded = np.argmax(y_pred_proba, axis=1)
    y_pred = y_pred_encoded - 1
    
    accuracy = accuracy_score(y_seq_encoded, y_pred_encoded)
    
    params_str = "layers=[LSTM(50), LSTM(50)], seq_len=60, epochs=20"
    metrics_str = f"Acc:{accuracy:.2%}"
    
    return model, params_str, metrics_str, y_pred[-1], y_pred_proba[-1]


def train_rnn(X, y, seq_length=60):
    """Train Simple RNN model"""
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create sequences - need to align y properly
    X_seq_list = []
    y_seq_list = []
    
    for i in range(len(X_scaled) - seq_length):
        X_seq_list.append(X_scaled[i:i+seq_length])
        y_seq_list.append(y[i+seq_length])  # Target is the label AFTER the sequence
    
    X_seq = np.array(X_seq_list)
    y_seq = np.array(y_seq_list)
    
    if len(X_seq) < 10:
        return None, "Insufficient data", "N/A", 0, 0.5
    
    # Encode labels: -1 -> 0, 0 -> 1, 1 -> 2
    y_seq_encoded = y_seq + 1
    
    # Build model
    model = Sequential([
        SimpleRNN(50, return_sequences=True, input_shape=(seq_length, X.shape[1])),
        Dropout(0.2),
        SimpleRNN(50, return_sequences=False),
        Dropout(0.2),
        Dense(3, activation='softmax')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Train
    model.fit(X_seq, y_seq_encoded, epochs=20, batch_size=32, verbose=0)
    
    # Predict
    y_pred_proba = model.predict(X_seq, verbose=0)
    y_pred_encoded = np.argmax(y_pred_proba, axis=1)
    y_pred = y_pred_encoded - 1
    
    accuracy = accuracy_score(y_seq_encoded, y_pred_encoded)
    
    params_str = "layers=[RNN(50), RNN(50)], seq_len=60, epochs=20"
    metrics_str = f"Acc:{accuracy:.2%}"
    
    return model, params_str, metrics_str, y_pred[-1], y_pred_proba[-1]


def monte_carlo_ml_simulation(df: pd.DataFrame, current_price: float, sims: int = 1000):
    """Monte Carlo simulation for ML prediction"""
    returns = df['Close'].pct_change().dropna().values
    
    if len(returns) < 30:
        return 0, 0.5
    
    mu = returns.mean()
    sigma = returns.std()
    
    # Simulate 5-day returns
    simulated_returns = np.random.normal(mu * 5, sigma * np.sqrt(5), sims)
    
    # Count how many exceed thresholds
    buy_count = np.sum(simulated_returns > 0.02)
    sell_count = np.sum(simulated_returns < -0.02)
    
    if buy_count > sell_count:
        prediction = 1
        confidence = buy_count / sims
    elif sell_count > buy_count:
        prediction = -1
        confidence = sell_count / sims
    else:
        prediction = 0
        confidence = 0.5
    
    params_str = f"sims={sims}, horizon=5d, normal dist"
    metrics_str = f"BUY prob:{buy_count/sims:.2%} SELL prob:{sell_count/sims:.2%}"
    
    return None, params_str, metrics_str, prediction, confidence


def run_ml_analysis(df: pd.DataFrame):
    """Run all ML models and return results"""
    results = []
    
    # Prepare features
    X, y, feature_names, returns_5d = prepare_ml_features(df, lookback=60)
    
    if X is None or len(X) < 100:
        st.error("Insufficient data for ML analysis. Need at least 100 data points with valid indicators.")
        return None
    
    current_price = df['Close'].iloc[-1]
    
    # 1. Random Forest
    with st.spinner("Training Random Forest..."):
        try:
            model, params, metrics, pred, proba = train_random_forest(X, y)
            rec = "BUY" if pred == 1 else "SELL" if pred == -1 else "HOLD"
            conf = np.max(proba) * 100
            results.append(["Random Forest", params, metrics, rec, f"{conf:.1f}%"])
        except Exception as e:
            results.append(["Random Forest", "Error", str(e), "N/A", "N/A"])
    
    # 2. XGBoost
    with st.spinner("Training XGBoost..."):
        try:
            model, params, metrics, pred, proba = train_xgboost(X, y)
            rec = "BUY" if pred == 1 else "SELL" if pred == -1 else "HOLD"
            conf = np.max(proba) * 100
            results.append(["XGBoost", params, metrics, rec, f"{conf:.1f}%"])
        except Exception as e:
            results.append(["XGBoost", "Error", str(e), "N/A", "N/A"])
    
    # 3. ARIMA + GARCH
    with st.spinner("Training ARIMA + GARCH..."):
        try:
            model, params, metrics, pred, conf = train_arima_garch(df)
            rec = "BUY" if pred == 1 else "SELL" if pred == -1 else "HOLD"
            results.append(["ARIMA + GARCH", params, metrics, rec, f"{conf*100:.1f}%"])
        except Exception as e:
            results.append(["ARIMA + GARCH", "Error", str(e), "N/A", "N/A"])
    
    # 4. LSTM
    with st.spinner("Training LSTM (this may take a minute)..."):
        try:
            model, params, metrics, pred, proba = train_lstm(X, y, seq_length=60)
            if model is not None:
                rec = "BUY" if pred == 1 else "SELL" if pred == -1 else "HOLD"
                conf = np.max(proba) * 100
                results.append(["LSTM", params, metrics, rec, f"{conf:.1f}%"])
            else:
                results.append(["LSTM", params, metrics, "N/A", "N/A"])
        except Exception as e:
            results.append(["LSTM", "Error", str(e), "N/A", "N/A"])
    
    # 5. RNN
    with st.spinner("Training RNN..."):
        try:
            model, params, metrics, pred, proba = train_rnn(X, y, seq_length=60)
            if model is not None:
                rec = "BUY" if pred == 1 else "SELL" if pred == -1 else "HOLD"
                conf = np.max(proba) * 100
                results.append(["RNN", params, metrics, rec, f"{conf:.1f}%"])
            else:
                results.append(["RNN", params, metrics, "N/A", "N/A"])
        except Exception as e:
            results.append(["RNN", "Error", str(e), "N/A", "N/A"])
    
    # 6. Monte Carlo
    with st.spinner("Running Monte Carlo simulation..."):
        try:
            model, params, metrics, pred, conf = monte_carlo_ml_simulation(df, current_price, sims=1000)
            rec = "BUY" if pred == 1 else "SELL" if pred == -1 else "HOLD"
            results.append(["Monte Carlo", params, metrics, rec, f"{conf*100:.1f}%"])
        except Exception as e:
            results.append(["Monte Carlo", "Error", str(e), "N/A", "N/A"])
    
    # Filter out invalid results
    valid_results = []
    for result in results:
        if len(result) >= 4:
            recommendation = result[3]
            confidence = result[4]
            if recommendation in ["BUY", "SELL", "HOLD"] and confidence != "N/A":
                valid_results.append(result)
    
    return valid_results if valid_results else None


def calculate_ensemble_recommendation(results, weights=None):
    """
    Calculate ensemble recommendation using weighted voting
    
    Args:
        results: List of ML results from run_ml_analysis
        weights: Dict of algorithm weights (default: equal weights)
    
    Returns:
        Tuple of (recommendation, confidence, agreement)
    """
    if not results:
        return "N/A", "0%", "N/A"
    
    # Default equal weights if not provided
    if weights is None:
        weights = {
            'Random Forest': 1.0/6,
            'XGBoost': 1.0/6,
            'ARIMA + GARCH': 1.0/6,
            'LSTM': 1.0/6,
            'RNN': 1.0/6,
            'Monte Carlo': 1.0/6
        }
    
    # Normalize weights to sum to 1.0
    total_weight = sum(weights.values())
    if total_weight > 0:
        weights = {k: v/total_weight for k, v in weights.items()}
    
    weighted_votes = {"BUY": 0.0, "SELL": 0.0, "HOLD": 0.0}
    confidences = []
    total_models = 0
    
    for result in results:
        algo_name = result[0]
        rec = result[3]
        conf_str = result[4]
        
        if rec in weighted_votes and algo_name in weights:
            weight = weights[algo_name]
            weighted_votes[rec] += weight
            total_models += 1
            
            try:
                conf_val = float(conf_str.replace("%", ""))
                confidences.append(conf_val)
            except:
                pass
    
    # Weighted majority
    ensemble_rec = max(weighted_votes, key=weighted_votes.get)
    
    # Average confidence
    avg_confidence = np.mean(confidences) if confidences else 0
    
    # Agreement percentage based on weighted votes
    max_vote = weighted_votes[ensemble_rec]
    agreement = (max_vote * 100) if total_models > 0 else 0
    
    # Count how many models voted for the winning recommendation
    vote_count = sum(1 for r in results if r[3] == ensemble_rec)
    
    return ensemble_rec, f"{avg_confidence:.1f}%", f"{agreement:.0f}% weighted agreement ({vote_count}/{total_models} models)"
