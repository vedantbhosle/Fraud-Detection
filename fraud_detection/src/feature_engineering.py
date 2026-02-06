import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
import os

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0 # Radius of Earth in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def main():
    print("Loading data...")
    # Determine base directory (fraud_detection/)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    
    df = pd.read_csv(os.path.join(DATA_DIR, "transactions.csv"))
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
    
    print("Generating Features...")
    
    # 1. Temporal Aggregations (Rolling Windows)
    # Count txns in last 1h and 24h
    df = df.set_index('timestamp')
    df['count_1h'] = df.groupby('user_id')['amount'].rolling('1h').count().values
    df['count_24h'] = df.groupby('user_id')['amount'].rolling('24h').count().values
    df = df.reset_index()
    
    # 2. Behavioral Ratios
    # Current Amount vs User's Avg Amount (Calculate cumulative mean safely)
    # Using expansion mean from pandas
    df['user_avg_amt'] = df.groupby('user_id')['amount'].transform(lambda x: x.expanding().mean())
    df['amt_ratio'] = df['amount'] / df['user_avg_amt']
    
    # 3. Geo-spatial Features (Distance from prev txn)
    # Shift location columns
    df['prev_lat'] = df.groupby('user_id')['location_lat'].shift(1)
    df['prev_long'] = df.groupby('user_id')['location_long'].shift(1)
    df['prev_time'] = df.groupby('user_id')['timestamp'].shift(1)
    
    # Time diff in minutes
    df['time_diff'] = (df['timestamp'] - df['prev_time']).dt.total_seconds() / 60.0
    df['time_diff'] = df['time_diff'].fillna(0)
    
    # Distance
    # Vectorized Haversine is hard without libs like geopy or creating complex numpy func.
    # Iterating is slow, but for 15k rows it's instant. Let's use numpy.
    
    def vectorized_haversine(lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        return 6371 * c

    # Handle NaNs in previous location (first txn) by filling with current location (dist=0)
    df['prev_lat'] = df['prev_lat'].fillna(df['location_lat'])
    df['prev_long'] = df['prev_long'].fillna(df['location_long'])
    
    df['dist_prev'] = vectorized_haversine(
        df['location_lat'], df['location_long'],
        df['prev_lat'], df['prev_long']
    )
    
    # Speed check (km/h)
    # Avoid divide by zero
    df['speed_kmh'] = df['dist_prev'] / ((df['time_diff']+0.01)/60) 
    
    # 4. Categorical Encoding
    # One-Hot Encoding for Category and Payment Method
    # We drop IDs for modeling
    df = pd.get_dummies(df, columns=['category', 'payment_method'], prefix=['cat', 'pay'], drop_first=True)
    
    # Drop UUIDs and raw leakage columns
    drop_cols = ['transaction_id', 'user_id', 'merchant_id', 'device_id', 'location_lat', 'location_long', 
                 'prev_lat', 'prev_long', 'prev_time', 'user_avg_amt']
    # Note: We keep is_fraud and fraud_type for training/analysis
    
    final_df = df.drop(columns=drop_cols)
    
    # Handle NaNs created by rolling/shifting
    final_df = final_df.fillna(0)
    
    output_path = os.path.join(DATA_DIR, "features.csv")
    final_df.to_csv(output_path, index=False)
    print(f"Features saved to {output_path} with shape {final_df.shape}")

if __name__ == "__main__":
    main()
