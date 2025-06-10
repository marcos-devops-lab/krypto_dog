# ai/model_trainer.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import xgboost as xgb
import joblib # For saving/loading models
import shap # For explainability
import os

# Assume data_loader and feature_engineer are in correct relative paths
from utils.data_loader import load_ohlcv
from ai.feature_engineer import generate_features

def generate_labels(df: pd.DataFrame, look_forward_candles: int = 5, profit_target_pct: float = 1.0, stop_loss_pct: float = 1.0) -> pd.Series:
    """
    Generates binary labels based on future price movement.
    1 if price moves up by profit_target_pct within look_forward_candles,
    0 if price moves down by stop_loss_pct within look_forward_candles,
    NaN otherwise (neutral/no clear signal).
    """
    labels = pd.Series(index=df.index, dtype=float)

    for i in range(len(df) - look_forward_candles):
        current_close = df['close'].iloc[i]
        future_period = df['close'].iloc[i+1 : i + look_forward_candles + 1]

        # Calculate max gain and max loss in the future period
        max_future_gain = (future_period.max() - current_close) / current_close * 100
        min_future_loss = (future_period.min() - current_close) / current_close * 100

        # Labeling logic
        if max_future_gain >= profit_target_pct:
            labels.iloc[i] = 1 # Profitable move expected
        elif min_future_loss <= -stop_loss_pct:
            labels.iloc[i] = 0 # Losing move expected (or adverse movement)
        # Else, leave as NaN (neutral outcomes to be dropped or handled differently)
    
    return labels

def train_model(config_path="config/strategy_config.yaml", save_model=True):
    # Load data path from config
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    data_path = config['data']['path']
    profit_target = config['exit_conditions']['take_profit'] # Use existing config values
    stop_loss = config['exit_conditions']['stop_loss']
    look_forward = config['exit_conditions']['holding_period'] # Use holding period as look forward window

    print(f"Loading data from {data_path}...")
    df_ohlcv = load_ohlcv(data_path)
    
    print("Generating features...")
    df_features = generate_features(df_ohlcv)

    print(f"Generating labels (look_forward={look_forward}, profit_target={profit_target}%, stop_loss={stop_loss}%)...")
    # Align labels to features DataFrame index
    labels = generate_labels(df_ohlcv, look_forward_candles=look_forward, 
                             profit_target_pct=profit_target, stop_loss_pct=stop_loss)
    
    # Merge features and labels, then drop rows with NaN labels
    combined_df = pd.concat([df_features, labels.rename('target')], axis=1).dropna()
    
    if combined_df.empty:
        print("Error: No data available after feature and label generation/dropna. Check data, indicators, and labeling parameters.")
        return

    X = combined_df.drop('target', axis=1)
    y = combined_df['target']

    # --- Data Splitting ---
    # Use a time-series split to avoid look-ahead bias.
    # Simple split: take first 80% for training, last 20% for testing.
    train_size = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Train label distribution:\n{y_train.value_counts(normalize=True)}")
    print(f"Test label distribution:\n{y_test.value_counts(normalize=True)}")

    if X_train.empty or X_test.empty or y_train.empty or y_test.empty:
        print("Not enough data for training/testing after splitting. Ensure you have sufficient historical data.")
        return

    # --- Model Training (XGBoost for this example) ---
    print("\nTraining XGBoost Classifier...")
    model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, random_state=42)
    model.fit(X_train, y_train)
    print("Model training complete.")

    # --- Model Evaluation ---
    print("\nEvaluating model on test set...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] # Probability of positive class

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # --- Feature Importance (via SHAP) ---
    print("\nCalculating SHAP values for feature importance...")
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        # shap.summary_plot(shap_values, X_test, plot_type="bar", show=False) # Requires matplotlib for plotting
        # For simplicity, just print top features by mean absolute SHAP value
        if isinstance(shap_values, list): # For multi-output models, often returns list
            shap_values = shap_values[1] # Take SHAP values for the positive class (label 1)
        
        shap_sum = np.abs(shap_values).mean(axis=0)
        importance_df = pd.DataFrame({'feature': X_test.columns, 'importance': shap_sum})
        importance_df = importance_df.sort_values(by='importance', ascending=False)
        print("\nTop 10 Feature Importance (Mean Absolute SHAP Value):")
        print(importance_df.head(10))
    except Exception as e:
        print(f"Could not calculate SHAP values: {e}. Ensure all dependencies are met (e.g., matplotlib for plotting).")

    # --- Save Model ---
    if save_model:
        model_path = "ai/trained_model.pkl"
        joblib.dump(model, model_path)
        print(f"\nTrained model saved to {model_path}")

    return model, X_test, y_test, y_pred, y_proba

if __name__ == "__main__":
    train_model()