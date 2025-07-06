import pandas as pd
import numpy as np
import os

# --- Configuration ---
ORIGINAL_DATA_FILE = "cleaned_sensor_data.csv"
MODIFIED_DATA_FILE = "cleaned_sensor_data_with_dummy_targets.csv"

# --- Dummy Data Ranges ---
SOIL_MOISTURE_MIN = 30.0
SOIL_MOISTURE_MAX = 70.0

LIGHT_INTENSITY_MIN = 1000.0
LIGHT_INTENSITY_MAX = 10000.0

GROWTH_FACTOR_MIN = 0.3
GROWTH_FACTOR_MAX = 1.0

# --- Load CSV ---
print(f"Loading data from {ORIGINAL_DATA_FILE}...")
try:
    df = pd.read_csv(ORIGINAL_DATA_FILE)
    print("Data loaded successfully.")
except FileNotFoundError:
    print(f"Error: {ORIGINAL_DATA_FILE} not found. Please ensure the CSV file is in the same directory.")
    exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

print("Generating dummy data for target features...")

# --- Generate Features ---
df['soil_moisture'] = np.random.uniform(SOIL_MOISTURE_MIN, SOIL_MOISTURE_MAX, size=len(df)).round(2)
df['light_intensity'] = np.random.uniform(LIGHT_INTENSITY_MIN, LIGHT_INTENSITY_MAX, size=len(df)).round(2)
df['growth_factor'] = np.random.uniform(GROWTH_FACTOR_MIN, GROWTH_FACTOR_MAX, size=len(df)).round(2)

# --- Growth Trigger Logic ---
# If soil_moisture > 40 and temperature > 25, then trigger growth
df['growth_trigger'] = ((df['soil_moisture'] > 40) & (df['temperature'] > 25)).astype(int)

# --- Show Sample ---
print("New columns added: 'soil_moisture', 'light_intensity', 'growth_factor', 'growth_trigger'")
print("First 5 rows of the modified data:")
print(df[['soil_moisture', 'light_intensity', 'growth_factor', 'growth_trigger']].head())

# --- Save to CSV ---
print(f"\nSaving modified data to {MODIFIED_DATA_FILE}...")
df.to_csv(MODIFIED_DATA_FILE, index=False)
print("✅ Modified data saved successfully.")

# --- Training Tips ---
print("\nTo use this modified data for training:")
print(f"1. Update the 'DATA_FILE' variable in your 'retrain_tdann_model_final.py' script to point to '{MODIFIED_DATA_FILE}'.")
print("2. Set: target_columns = ['soil_moisture', 'light_intensity', 'growth_factor', 'growth_trigger']")
print("3. If 'N', 'P', and 'K' exist, you can also compute and add:")
print("     df['npk_sum'] = df['N'] + df['P'] + df['K']")
print("     target_columns.append('npk_sum')")
print("4. Run your model retraining script.")
