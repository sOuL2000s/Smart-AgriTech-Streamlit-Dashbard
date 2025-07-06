import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os

# --- Constants & Configuration ---
DATA_FILE = "cleaned_sensor_data.csv"
MODEL_SAVE_PATH = "tdann_pnsm_model.keras"
LOOKBACK_WINDOW = 5
EPOCHS = 50 # Increased epochs for better training potentially
BATCH_SIZE = 16
VALIDATION_SPLIT = 0.2

# --- Load & Prepare Data ---
try:
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded {len(df)} rows from {DATA_FILE}")
except FileNotFoundError:
    print(f"❌ Error: {DATA_FILE} not found. Please ensure Phase 2 has generated this file.")
    exit() # Exit if data file is not found
except Exception as e:
    print(f"❌ Error loading data: {e}")
    exit()

# Convert timestamp and set as index (optional for sequence creation, but good practice)
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
df = df.dropna(subset=['timestamp']) # Drop rows with invalid timestamps
df = df.sort_values('timestamp') # Ensure data is time-ordered

# Drop original 'ph' if 'pH' is the correct one used consistently in dashboard/sim
# Or, if both exist and are distinct, keep them and decide which to use.
# For consistency with the dashboard and `insert-sample-data.py` I'll assume 'pH' is preferred.
if 'ph' in df.columns and 'pH' in df.columns and 'ph' != 'pH':
    # If both exist and are different, you might want to choose one or combine them
    print("Warning: Both 'ph' and 'pH' columns exist. Using 'pH' and dropping 'ph'.")
    df = df.drop(columns=['ph'])
elif 'ph' in df.columns and 'pH' not in df.columns:
    print("Renaming 'ph' to 'pH' for consistency.")
    df = df.rename(columns={'ph': 'pH'})

# Convert all relevant columns to numeric, coercing errors to NaN
# This is crucial as Firebase data might come in as strings
numeric_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'pH', 'rainfall',
                'soil_moisture', 'light_intensity', 'growth_factor', 'growth_trigger']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Handle missing values: fill with forward fill then back fill to handle initial NaNs
# This is a simple strategy; more advanced imputation might be needed for real data
df = df.fillna(method='ffill').fillna(method='bfill')
if df.isnull().sum().sum() > 0:
    print("Warning: Some NaN values remain after fillna. Consider more robust imputation if data is sparse.")
    # For now, drop rows with any remaining NaNs if critical features are missing
    df = df.dropna(subset=numeric_cols + ['label', 'crop_stage'])


# One-hot encode categorical features: 'label' (crop type) and 'crop_stage'
# Ensure df['label'] is correctly populated from 'cleaned_sensor_data.csv'
if 'label' in df.columns:
    crop_dummies = pd.get_dummies(df['label'], prefix='crop')
    df = pd.concat([df, crop_dummies], axis=1)
    print(f"One-hot encoded {len(crop_dummies.columns)} crop types.")
else:
    print("Warning: 'label' column not found for crop one-hot encoding.")
    crop_dummies = pd.DataFrame() # Create empty if not found

if 'crop_stage' in df.columns:
    # Handle potential non-string values by converting to string before get_dummies
    df['crop_stage'] = df['crop_stage'].astype(str) 
    stage_dummies = pd.get_dummies(df['crop_stage'], prefix='stage')
    df = pd.concat([df, stage_dummies], axis=1)
    print(f"One-hot encoded {len(stage_dummies.columns)} crop stages.")
else:
    print("Warning: 'crop_stage' column not found for stage one-hot encoding.")
    stage_dummies = pd.DataFrame() # Create empty if not found

# Define Input Features for the TDANN model
base_sensor_features = ['N', 'P', 'K', 'temperature', 'humidity', 'pH', 'rainfall']
# Add new biological signals as input features
biological_features = ['growth_factor', 'growth_trigger'] # Ensure these exist in your cleaned_sensor_data.csv
# Combine all input features
input_features = [col for col in base_sensor_features + biological_features + crop_dummies.columns.tolist() + stage_dummies.columns.tolist() if col in df.columns]

# Define Output Targets
# For "N, P, K balance", we'll create a simple sum as a proxy for general nutrient levels/health
df['nutrient_sum_proxy'] = df['N'] + df['P'] + df['K']
output_targets = ['soil_moisture', 'light_intensity', 'nutrient_sum_proxy']

# Filter DataFrame to only include necessary columns for scaling and sequence creation
df_model_input = df[input_features + output_targets].copy()

# Drop rows with any remaining NaNs in the selected features/targets
df_model_input = df_model_input.dropna()

if df_model_input.empty:
    print("❌ Error: No complete data rows available after preprocessing for model training.")
    exit()

print(f"DataFrame for model training has {len(df_model_input)} rows and {len(df_model_input.columns)} columns.")

# Separate features (X) and targets (Y)
X_data_raw = df_model_input[input_features].values
y_data_raw = df_model_input[output_targets].values

# --- Scaling ---
# Scale input features
input_scaler = MinMaxScaler()
X_scaled = input_scaler.fit_transform(X_data_raw)

# Scale output targets
output_scaler = MinMaxScaler()
y_scaled = output_scaler.fit_transform(y_data_raw)

print(f"Input features scaled. Shape: {X_scaled.shape}")
print(f"Output targets scaled. Shape: {y_scaled.shape}")

# --- Time-Delay Setup (TDANN: lookback window) ---
def create_sequences(input_data, output_data, lookback):
    """
    Creates sequences for TDANN (LSTM) model.
    :param input_data: Scaled input features (2D numpy array)
    :param output_data: Scaled target outputs (2D numpy array)
    :param lookback: Number of previous timesteps to consider for prediction
    :return: X (input sequences), y (corresponding targets)
    """
    X, y = [], []
    for i in range(lookback, len(input_data)):
        X.append(input_data[i - lookback:i])
        y.append(output_data[i]) # Append the entire row of targets for the current timestep
    return np.array(X), np.array(y)

X, y = create_sequences(X_scaled, y_scaled, LOOKBACK_WINDOW)

print(f"Created sequences. X shape: {X.shape}, y shape: {y.shape}")

# --- TDANN Model (LSTM-based) ---
model = Sequential()
# Input shape: (timesteps, features) -> (LOOKBACK_WINDOW, number_of_input_features)
model.add(LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(32)) # No return_sequences=True for the last LSTM layer before Dense
model.add(Dropout(0.2)) # Added another Dropout layer
# Final Dense layer for multi-output prediction
model.add(Dense(len(output_targets))) # Output: [moisture_pred, light_pred, nutrient_pred]

model.compile(optimizer='adam', loss='mse')
model.summary()

# --- Model Training ---
print("\n--- Training Model ---")
history = model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT, shuffle=False)

# --- Plot Loss ---
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('TDANN Training and Validation Loss (Multi-Output)')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error (MSE)')
plt.legend()
plt.grid(True)
plt.show()

# --- Save Model ---
print(f"\nSaving model to {MODEL_SAVE_PATH}...")
model.save(MODEL_SAVE_PATH)
print("Model saved successfully.")

# --- Optional: Make a prediction and inverse transform to see raw values ---
if X.shape[0] > 0:
    sample_input = X[-1].reshape(1, X.shape[1], X.shape[2]) # Get last sequence
    predicted_scaled = model.predict(sample_input)
    predicted_raw = output_scaler.inverse_transform(predicted_scaled)
    actual_raw = output_scaler.inverse_transform(y[-1].reshape(1, -1))

    print("\n--- Sample Prediction ---")
    print(f"Predicted (Raw): Soil Moisture={predicted_raw[0][0]:.2f}%, "
          f"Light Intensity={predicted_raw[0][1]:.2f}, "
          f"Nutrient Sum Proxy={predicted_raw[0][2]:.2f}")
    print(f"Actual (Raw):    Soil Moisture={actual_raw[0][0]:.2f}%, "
          f"Light Intensity={actual_raw[0][1]:.2f}, "
          f"Nutrient Sum Proxy={actual_raw[0][2]:.2f}")
else:
    print("Not enough data to make a sample prediction.")

# You might want to save the scalers as well if you plan to use them for live predictions
# from joblib import dump, load
# dump(input_scaler, 'input_scaler.joblib')
# dump(output_scaler, 'output_scaler.joblib')