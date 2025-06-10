import sys
import os

# Add the project root to the Python path
# This assumes model_trainer.py is in krypto_dog/ai/
# And ml/feature_engineer.py is in krypto_dog/ml/
# The project root is krypto_dog/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# The rest of your imports start here
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib
import yaml
import logging
# ... (rest of your existing imports) ...

# Import the feature engineering module (this line stays the same)
from ml.feature_engineer import load_data, add_technical_indicators, add_lagged_features, \
                                add_rolling_features, generate_labels, preprocess_features

# ... (rest of your model_trainer.py code) ...
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib
import yaml
import logging
import os

# Import the feature engineering module
# Ensure ml/feature_engineer.py is in the correct path relative to this script's execution
try:
    from ml.feature_engineer import load_data, add_technical_indicators, add_lagged_features, \
                                    add_rolling_features, generate_labels, preprocess_features
except ImportError as e:
    logging.error(f"Failed to import feature_engineer module: {e}. "
                  "Ensure ml/feature_engineer.py exists and is accessible.")
    raise

# Set up logging for better feedback
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_config(config_path='config/strategy_config.yaml'):
    """Loads configuration from a YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logging.error(f"Config file not found at {config_path}. Please ensure it exists.")
        return None
    except yaml.YAMLError as e:
        logging.error(f"Error parsing config file: {e}")
        return None

def train_model(config_path='config/strategy_config.yaml', model_params=None):
    """
    Orchestrates the data loading, feature engineering, labeling,
    XGBoost model training, evaluation, and saving.

    :param config_path: Path to the main configuration YAML file.
    :param model_params: Dictionary of XGBoost hyperparameters to use.
                         If None, default params from config will be used.
    """
    logging.info("Starting model training process...")

    config = get_config(config_path)
    if config is None:
        logging.error("Configuration could not be loaded. Aborting training.")
        return

    data_config = config.get('data_fetching', {})
    model_config = config.get('model_training', {})
    
    data_file_path = os.path.join('data', data_config.get('csv_file', 'PEPE_1m.csv')) # Use os.path.join for robustness

    # 1. Load Data
    df_ohlcv = load_data(data_file_path)
    if df_ohlcv is None or df_ohlcv.empty:
        logging.error("Failed to load OHLCV data. Aborting training.")
        return

    # 2. Feature Engineering
    logging.info("Generating features...")
    df_features = add_technical_indicators(df_ohlcv.copy())
    df_features = add_lagged_features(df_features)
    df_features = add_rolling_features(df_features)

    # Access labeling parameters correctly from the nested config structure
    logging.info(f"Generating labels (look_forward_candles={model_config['labeling']['look_forward_candles']}, "
                 f"profit_target_pct={model_config['labeling']['profit_target_pct']}%, "
                 f"stop_loss_pct={model_config['labeling']['stop_loss_pct']}%)...")
    
    # This line stays as is, as generate_labels expects the model_config dictionary
    df_labeled = generate_labels(df_features, model_config)

    # 4. Preprocess Features and Labels
    # Separate features (X) and target (y)
    # Ensure 'label' is the target. Drop other columns that are not features (like raw OHLCV if desired).
    # The preprocess_features function will handle NaNs and scaling, returning the feature matrix and scaler
    df_processed, scaler = preprocess_features(df_labeled.copy())

    # Ensure 'label' column exists and is numeric after preprocessing
    if 'label' not in df_processed.columns:
        logging.error("Label column 'label' not found after preprocessing. Aborting training.")
        return
    
    # Extract features (X) and target (y)
    # Exclude non-feature columns including the raw OHLCV if they were concat'd back
    # IMPORTANT: The feature_columns list from preprocess_features should be used here to ensure consistency
    # Let's adjust preprocess_features to return feature_columns directly for clarity.
    # For now, we'll re-derive them, assuming 'label' and 'OHLCV' are not features.
    feature_columns = [col for col in df_processed.columns if col not in ['label', 'open', 'high', 'low', 'close', 'volume']]
    
    X = df_processed[feature_columns]
    y = df_processed['label']

    if X.empty or y.empty:
        logging.warning("No data or labels remaining after preprocessing. Model training skipped.")
        return

    # 5. Split Data (Train/Test)
    test_size = model_config.get('test_size', 0.25)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
    logging.info(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
    logging.info("Train label distribution:\n" + y_train.value_counts(normalize=True).to_string())
    logging.info("Test label distribution:\n" + y_test.value_counts(normalize=True).to_string())

    # 6. Define XGBoost Model Parameters
    # Default parameters - these can be overridden by 'model_params' from Optuna
    xgb_default_params = {
        'objective': 'binary:logistic', # For binary classification
        'eval_metric': 'logloss',       # Evaluation metric during training
        'use_label_encoder': False,     # Suppress warning for newer XGBoost versions
        'n_estimators': 500,            # Number of boosting rounds
        'learning_rate': 0.05,          # Step size shrinkage
        'max_depth': 5,                 # Maximum depth of a tree
        'subsample': 0.7,               # Subsample ratio of the training instance
        'colsample_bytree': 0.7,        # Subsample ratio of columns when constructing each tree
        'gamma': 0.1,                   # Minimum loss reduction required to make a further partition
        'lambda': 1,                    # L2 regularization term on weights
        'alpha': 0,                     # L1 regularization term on weights
        'random_state': 42,
        'tree_method': 'hist',          # For faster training on larger datasets
        # Crucial for imbalanced datasets: balance positive and negative weights.
        # It's (count_negative_instances / count_positive_instances)
        'scale_pos_weight': (y_train.value_counts()[0] / y_train.value_counts()[1]) if y_train.value_counts()[1] != 0 else 1
    }

    # Override defaults with any parameters provided by Optuna
    if model_params:
        xgb_default_params.update(model_params)
        logging.info("XGBoost model parameters overridden by Optuna suggestions.")
    else:
        logging.info("Using default XGBoost model parameters.")

    # 7. Train XGBoost Classifier
    logging.info("Training XGBoost Classifier...")
    model = xgb.XGBClassifier(**xgb_default_params)
    model.fit(X_train, y_train)
    logging.info("Model training complete.")

    # 8. Evaluate Model
    logging.info("Evaluating model on test set...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] # Probability of the positive class

    logging.info("\nClassification Report:\n" + classification_report(y_test, y_pred))
    logging.info(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
    logging.info("Confusion Matrix:\n" + str(confusion_matrix(y_test, y_pred)))

    # 9. Feature Importance (using SHAP - requires shap to be installed and working)
    try:
        import shap
        logging.info("Calculating SHAP values for feature importance...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        # Ensure shap_values is an array, not a list of arrays (for binary classification, it's a list)
        if isinstance(shap_values, list) and len(shap_values) > 1:
            # For binary classification, shap_values[1] typically corresponds to the positive class
            shap_values = shap_values[1]

        # Calculate mean absolute SHAP value for each feature
        feature_importance = pd.DataFrame({
            'feature': X_test.columns,
            'importance': np.abs(shap_values).mean(axis=0)
        }).sort_values(by='importance', ascending=False)
        logging.info("\nTop 10 Feature Importance (Mean Absolute SHAP Value):\n" + feature_importance.head(10).to_string())
    except ImportError:
        logging.warning("SHAP library not installed. Skipping SHAP feature importance calculation.")
    except Exception as e:
        logging.warning(f"Error calculating SHAP values: {e}. Skipping SHAP feature importance.")

    # 10. Save Model and Scaler
    model_output_dir = 'ai'
    os.makedirs(model_output_dir, exist_ok=True)
    model_save_path = os.path.join(model_output_dir, 'trained_model.pkl')
    scaler_save_path = os.path.join(model_output_dir, 'scaler.pkl') # Save scaler for live prediction

    joblib.dump(model, model_save_path)
    joblib.dump(scaler, scaler_save_path) # Save the scaler used for preprocessing
    logging.info(f"Trained model saved to {model_save_path}")
    logging.info(f"Feature scaler saved to {scaler_save_path}")

    # Return performance metrics for Optuna
    return {
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'precision_0': 0 if '0' not in classification_report(y_test, y_pred, output_dict=True)['0'] else classification_report(y_test, y_pred, output_dict=True)['0']['precision'],
        'recall_0': 0 if '0' not in classification_report(y_test, y_pred, output_dict=True)['0'] else classification_report(y_test, y_pred, output_dict=True)['0']['recall'],
        'f1_0': 0 if '0' not in classification_report(y_test, y_pred, output_dict=True)['0'] else classification_report(y_test, y_pred, output_dict=True)['0']['f1-score'],
        'total_samples': X_test.shape[0]
    }


if __name__ == "__main__":
    # This block runs when model_trainer.py is executed directly for testing
    # It will use the default config/strategy_config.yaml
    train_model()