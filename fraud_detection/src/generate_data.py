import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta
import uuid
import os

fake = Faker()
Faker.seed(42)
np.random.seed(42)
random.seed(42)

# --- Configuration ---
NUM_USERS = 1000
NUM_MERCHANTS = 50
SIMULATION_DAYS = 30
START_DATE = datetime(2024, 1, 1)

# Fraud probabilities (per day or per check)
FRAUD_INJECTION_RATE = 0.05  # Target roughly 5% fraud

class User:
    def __init__(self):
        self.user_id = str(uuid.uuid4())[:8]
        self.name = fake.name()
        # Simple location simulation: (lat, lon)
        loc = fake.local_latlng(country_code='US')
        self.home_lat = float(loc[0])
        self.home_lon = float(loc[1])
        self.avg_txn_amount = np.random.lognormal(mean=3, sigma=1) 
        self.device_id = fake.sha256()[:16] # Static device ID for user (simulates effective device fingerprinting)
    
    def get_location(self):
        return (self.home_lat, self.home_lon)

class Merchant:
    def __init__(self):
        self.merchant_id = str(uuid.uuid4())[:8]
        self.name = fake.company()
        self.category = random.choice(['retail', 'dining', 'travel', 'grocery', 'digital', 'gas'])
        self.location = fake.local_latlng(country_code='US')

    def get_location(self):
         return (float(self.location[0]), float(self.location[1]))

def haversine_distance(coord1, coord2):
    R = 6371
    lat1, lon1 = np.radians(coord1)
    lat2, lon2 = np.radians(coord2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

print("Initializing entities...")
users = [User() for _ in range(NUM_USERS)]
merchants = [Merchant() for _ in range(NUM_MERCHANTS)]

transactions = []

def add_txn(user, merchant, time, amount, is_fraud=0, type="normal", location=None):
    if location is None:
        location = merchant.get_location()
        
    # Add random jitter to time (0-59 seconds)
    actual_time = time + timedelta(seconds=random.randint(0, 59))
        
    transactions.append({
        "transaction_id": str(uuid.uuid4()),
        "user_id": user.user_id,
        "merchant_id": merchant.merchant_id,
        "amount": round(amount, 2),
        "timestamp": actual_time,
        "location_lat": location[0],
        "location_long": location[1],
        "category": merchant.category,
        "payment_method": random.choice(['credit_card', 'debit_card', 'digital_wallet']),
        "device_id": user.device_id if random.random() > 0.1 else fake.sha256()[:16], # 10% chance of new device
        "is_fraud": is_fraud,
        "fraud_type": type if is_fraud else "none"
    })

print(f"Simulating {SIMULATION_DAYS} days...")
current_time = START_DATE
end_time = START_DATE + timedelta(days=SIMULATION_DAYS)

while current_time < end_time:
    # Increment time by random small intervals (e.g., 5-30 mins)
    step_minutes = random.randint(5, 30)
    current_time += timedelta(minutes=step_minutes)
    
    # 1. Normal Activity
    # Only a few users transact at any given moment
    active_users = random.sample(users, k=random.randint(1, 10)) 
    
    for user in active_users:
        # Pick a merchant
        merchant = random.choice(merchants)
        
        # Amount based on user habit
        amt = abs(np.random.normal(user.avg_txn_amount, user.avg_txn_amount*0.2))
        amt = max(1.0, amt)
        
        add_txn(user, merchant, current_time, amt, is_fraud=0)

    # 2. Fraud Injection Logic
    if random.random() < 0.15: # Significantly increased chance
        fraud_type = random.choice(['velocity', 'amount_spike', 'impossible_travel'])
        victim = random.choice(users)
        
        if fraud_type == 'velocity':
            # Subtler velocity: 2-5 txns (harder to distinguish)
            merchant = random.choice(merchants)
            for i in range(random.randint(2, 5)):
                t = current_time + timedelta(seconds=random.randint(1, 600))
                amt = random.uniform(10, 50)
                add_txn(victim, merchant, t, amt, is_fraud=1, type="velocity")
                
        elif fraud_type == 'amount_spike':
            # Smaller spike: 3x-10x overlap
            merchant = random.choice(merchants)
            amt = victim.avg_txn_amount * random.uniform(3, 10) 
            add_txn(victim, merchant, current_time, amt, is_fraud=1, type="amount_spike")
            
        elif fraud_type == 'impossible_travel':
            # 1 normal txn now, 1 fraud txn far away in 10 mins
            # First, normal txn
            m1 = random.choice(merchants)
            add_txn(victim, m1, current_time, victim.avg_txn_amount, is_fraud=0)
            
            # Second, far away txn
            # Force a merchant that is far? Or just fake the location.
            # Let's fake the location to be user's antipodes or just far
            far_lat = -victim.home_lat
            far_lon = -victim.home_lon
            t2 = current_time + timedelta(minutes=15)
            m2 = random.choice(merchants)
            
            add_txn(victim, m2, t2, victim.avg_txn_amount*2, is_fraud=1, type="impossible_travel", location=(far_lat, far_lon))

print("Compiling DataFrame...")
df = pd.DataFrame(transactions)
df = df.sort_values('timestamp').reset_index(drop=True)

# Validation Stats
print(f"Total Transactions: {len(df)}")
print(f"Fraud Rate: {df['is_fraud'].mean():.2%}")
print("\nFraud Breakdown:")
print(df[df['is_fraud']==1]['fraud_type'].value_counts())

# Save
# Determine base directory (fraud_detection/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True) # Ensure dir exists

output_path = os.path.join(DATA_DIR, "transactions.csv")
df.to_csv(output_path, index=False)
print(f"Saved to {output_path}")
