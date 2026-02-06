import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, average_precision_score, precision_recall_curve, confusion_matrix
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt
import os
import pickle

def main():
    print("Loading features...")
    # Determine base directory (fraud_detection/)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    MODEL_DIR = os.path.join(BASE_DIR, "model")
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = pd.read_csv(os.path.join(DATA_DIR, "features.csv"))
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    # Separate Features and Target
    # fraud_type is for analysis, not input
    X = df.drop(columns=['is_fraud', 'fraud_type', 'timestamp'])
    y = df['is_fraud']
    
    # Time-based Split (Train on first 80%, Test on last 20%)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"Train Fraud Rate: {y_train.mean():.2%}")
    print(f"Test Fraud Rate: {y_test.mean():.2%}")
    
    # Model Training
    # Calculate scale_pos_weight estimate
    ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)
    
    print(f"Training XGBoost (scale_pos_weight={ratio:.1f})...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=ratio,
        eval_metric='logloss',
        use_label_encoder=False
    )
    
    model.fit(X_train, y_train)
    
    # Evaluation
    print("Evaluating...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    print("\n--- Model Performance ---")
    print(classification_report(y_test, y_pred))
    
    pr_auc = average_precision_score(y_test, y_pred_proba)
    print(f"PR-AUC Score: {pr_auc:.4f}")
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:\n{cm}")
    
    # Feature Importance (SHAP)
    print("\nCalculating SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # Save Feature Importance Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    shap_path = os.path.join(MODEL_DIR, "shap_summary.png")
    plt.savefig(shap_path)
    print(f"SHAP Summary plot saved to {shap_path}")
    
    # Save Model (JSON)
    model_path = os.path.join(MODEL_DIR, "fraud_xgboost.json")
    model.save_model(model_path)
    print(f"Model saved to {model_path}")

    # Save Model (PKL)
    pkl_path = os.path.join(MODEL_DIR, "fraud_xgboost.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {pkl_path}")

if __name__ == "__main__":
    main()
