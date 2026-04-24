# =========================================
# STEP 3: SIMULATE MULTIPLE BANKS
# =========================================

import pandas as pd
import numpy as np

# Load your dataset again
df = pd.read_csv("data/upi_synthetic_dataset.csv")

# Shuffle dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split into 3 banks
bank_A = df.iloc[:int(0.33 * len(df))]
bank_B = df.iloc[int(0.33 * len(df)):int(0.66 * len(df))]
bank_C = df.iloc[int(0.66 * len(df)):]

# Save datasets
bank_A.to_csv("data/bank_A.csv", index=False)
bank_B.to_csv("data/bank_B.csv", index=False)
bank_C.to_csv("data/bank_C.csv", index=False)

print("Banks Created Successfully!")
print("Bank A:", bank_A.shape)
print("Bank B:", bank_B.shape)
print("Bank C:", bank_C.shape)