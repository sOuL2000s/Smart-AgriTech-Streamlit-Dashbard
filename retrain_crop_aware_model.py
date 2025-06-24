import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# --- Load Your Full Crop Dataset ---
df = pd.read_csv("cleaned_sensor_data.csv")

# --- Ensure Timestamp Index ---
if 'timestamp' not in df.columns:
    df['timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='h')
df.set_index('timestamp', inplace=True)

# --- One-Hot Encode Crop Labels ---
df['label'] = df['label'].str.lower().str.strip()
crop_dummies = pd.get_dummies(df['label'], prefix='crop')
df = pd.concat([df, crop_dummies], axis=1)

# --- Final Feature Columns ---
base_features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
final_features = base_features + crop_dummies.columns.tolist()

# --- Scaling ---
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df[final_features])

# --- Sequence Generation ---
def create_sequences(data, lookback=5):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        y.append(data[i, base_features.index('temperature')])  # Predicting temperature as example
    return np.array(X), np.array(y)

lookback = 5
X, y = create_sequences(data_scaled, lookback)

# --- Build Model ---
model = Sequential()
model.add(Input(shape=(X.shape[1], X.shape[2])))
model.add(LSTM(64, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(32))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# --- Train ---
history = model.fit(X, y, epochs=30, batch_size=16, validation_split=0.2)

# --- Save ---
model.save("tdann_pnsm_model.keras")
print("✅ New model trained and saved as tdann_pnsm_model.keras")

# --- Optional: Plot Loss ---
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Training Loss")
plt.legend()
plt.show()
