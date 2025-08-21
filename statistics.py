# Run this script to get the exact statistics from your training data
import pandas as pd
import numpy as np

# Load the data
data = pd.read_excel("Concrete_Data.xls")

# Split features
X = data.drop(columns=['Concrete compressive strength(MPa, megapascals) '])

# Calculate the engineered features exactly as in your notebook
cement = X['Cement (component 1)(kg in a m^3 mixture)']
blast_slag = X['Blast Furnace Slag (component 2)(kg in a m^3 mixture)']
fly_ash = X['Fly Ash (component 3)(kg in a m^3 mixture)']
water = X['Water  (component 4)(kg in a m^3 mixture)']

# Calculate total cementitious
total_cementitious = cement + blast_slag + fly_ash

# Calculate water cement ratio
water_cement_ratio = water / (cement + 1e-5)

# Print the exact statistics
print("Copy these values to your app.py:")
print("TRAINING_STATS = {")
print(f"    'total_cementitious_mean': {total_cementitious.mean():.3f},")
print(f"    'total_cementitious_std': {total_cementitious.std():.3f},")
print(f"    'water_cement_ratio_mean': {water_cement_ratio.mean():.3f},")
print(f"    'water_cement_ratio_std': {water_cement_ratio.std():.3f},")
print("}")

# Also check the specific test sample values
test_sample_total_cem = 332.0 + 142.5 + 0.0  # 474.5
test_sample_wcr = 228.0 / 332.0  # 0.6867

print(f"\nTest sample total_cementitious: {test_sample_total_cem}")
print(f"Test sample water_cement_ratio: {test_sample_wcr:.4f}")
print(f"Is total_cem < mean? {test_sample_total_cem < total_cementitious.mean()}")
print(f"Abnormal factor for test: {abs((test_sample_wcr - water_cement_ratio.mean()) / water_cement_ratio.std()):.3f}")