import pandas as pd
import numpy as np

np.random.seed(42)

n = 5000

data = pd.DataFrame({
    "transaction_id": np.arange(1, n+1),
    "utr_number": ["UPI" + str(np.random.randint(10000000, 99999999)) for _ in range(n)],
    "amount": np.round(np.random.exponential(scale=2000, size=n), 2),
    "timestamp": pd.date_range(start="2026-01-01", end="2026-04-20", periods=n),
    "sender_id": np.random.randint(10000, 20000, n),
    "receiver_id": np.random.randint(20000, 30000, n),
    "device_type": np.random.choice(["mobile", "web"], n),
    "upi_app": np.random.choice(["GPay", "PhonePe", "Paytm"], n),
    "location": np.random.choice(["Bangalore", "Mumbai", "Delhi", "Chennai"], n),
    "is_night": np.random.choice([0, 1], n),
    "transaction_velocity": np.round(np.random.rand(n), 3),
})

data["is_fraud"] = (
    (data["amount"] > 5000) &
    (data["is_night"] == 1) &
    (data["transaction_velocity"] > 0.8)
).astype(int)

data.to_csv("upi_synthetic_dataset.csv", index=False)
print("Dataset created with UTR numbers!")