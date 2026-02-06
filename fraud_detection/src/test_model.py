import pickle
import pandas as pd
import numpy as np
import os
import random

def main():
    # Setup Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH = os.path.join(BASE_DIR, "model", "fraud_xgboost.pkl")
    DATA_PATH = os.path.join(BASE_DIR, "data", "features.csv")
    
    print(f"Loading model from {MODEL_PATH}...")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
        
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    # Drop non-feature columns for prediction
    # We need to keep 'is_fraud' for validation, but drop it for inference
    feature_cols = [c for c in df.columns if c not in ['is_fraud', 'fraud_type', 'timestamp']]
    
    # Select Samples: 5 Fraud, 5 Normal
    frauds = df[df['is_fraud'] == 1].sample(5)
    normals = df[df['is_fraud'] == 0].sample(5)
    samples = pd.concat([frauds, normals]).sample(frac=1) # Shuffle
    
    print(f"\nRunning Inference on {len(samples)} random samples...\n")
    print(f"{'ID':<5} | {'Actual':<10} | {'Pred':<10} | {'Conf%':<6} | {'Result':<10}")
    print("-" * 60)
    
    for idx, row in samples.iterrows():
        # Prepare input vector (must match training columns exactly)
        input_data = pd.DataFrame([row[feature_cols]])
        
        # Predict
        prob = model.predict_proba(input_data)[0][1]
        pred = 1 if prob > 0.5 else 0
        
        # Display
        actual = int(row['is_fraud'])
        status = "✅ CORRECT" if actual == pred else "❌ MISS"
        
        # Color code confidence
        conf = prob if pred == 1 else 1-prob
        
        print(f"{idx:<5} | {actual:<10} | {pred:<10} | {conf:.1%} | {status}")
        
    print("\n" + "="*60)
    print("Testing Manual Scenarios (Synthetic Data)")
    print("="*60)
    print("NOTE: 'cat_digital' and 'pay_credit_card' are Reference Categories (all 0s)")

    # Helper to create a single-row DF
    def create_sample(base_cols, updates):
        # Start with all zeros (which implies Reference Category for One-Hot)
        data = {col: 0.0 for col in base_cols}
        # Update specific features
        for k, v in updates.items():
            if k in data:
                data[k] = float(v)
        return pd.DataFrame([data])

    # Scenario 1: Completely Normal Transaction
    # - Amount $20 (approx average)
    # - 1 current transaction (count_1h=1 is minimum possible)
    # - Ratio 1.0
    # - Retail / Credit Card (Reference for Pay, 1 for Retail)
    normal_sample = create_sample(feature_cols, {
        'amount': 20.0,
        'count_1h': 1, 
        'count_24h': 1,
        'amt_ratio': 1.0,
        'speed_kmh': 0.0,
        'dist_prev': 0.0,
        'cat_retail': 1
        # pay_credit_card is reference, so we leave pay_debit/pay_wallet as 0
    })

    # Scenario 2: High Velocity Fraud
    # - Low amount ($20)
    # - BUT 15 transactions in last hour!
    # - Digital / Debit Card
    velocity_fraud = create_sample(feature_cols, {
        'amount': 20.0,
        'count_1h': 15,     # <--- VERY SUSPICIOUS
        'count_24h': 20,
        'amt_ratio': 1.0,
        'speed_kmh': 20.0,
        'pay_debit_card': 1
        # cat_digital is reference, so we leave cat_* as 0
    })

    # Scenario 3: Impossible Travel
    # - Normal amount
    # - Speed is 5000 km/h
    # - Retail (Should not be moving this fast!)
    travel_fraud = create_sample(feature_cols, {
        'amount': 50.0,
        'count_1h': 2,
        'amt_ratio': 1.0,
        'dist_prev': 1200.0,
        'time_diff': 15.0,    # 15 mins
        'speed_kmh': 4800.0,  # <--- SUSPICIOUS (1200km / 0.25h)
        'cat_retail': 1
    })

    # Scenario 4: Amount Spike
    # - Amount is 50x ratio
    spike_fraud = create_sample(feature_cols, {
        'amount': 5000.0,     # <--- SUSPICIOUS
        'count_1h': 1,
        'amt_ratio': 50.0,    # <--- SUSPICIOUS
        'speed_kmh': 0.0,
        'cat_dining': 1
    })

    scenarios = [
        ("Normal User", normal_sample),
        ("Velocity Attack", velocity_fraud),
        ("Impossible Travel", travel_fraud),
        ("Amount Spike", spike_fraud)
    ]

    print(f"{'Scenario':<20} | {'Pred':<10} | {'Conf%':<6}")
    print("-" * 45)

    for name, sample_df in scenarios:
        # Align columns
        sample_df = sample_df[feature_cols]
        
        prob = model.predict_proba(sample_df)[0][1]
        pred = "FRAUD" if prob > 0.5 else "NORMAL"
        
        # Color output slightly for readability if supported, else plain
        print(f"{name:<20} | {pred:<10} | {prob:.1%}")

    print("\nTest Complete.")

if __name__ == "__main__":
    main()
