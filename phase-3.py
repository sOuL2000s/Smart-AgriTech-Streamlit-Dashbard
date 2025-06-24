# Phase 3: TDANN + PNSM (Plant Growth Prediction Model)

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# --- Load & Prepare Data ---
df = pd.read_csv("cleaned_sensor_data.csv")  # Exported from Phase 2
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp')

features = ['soil_moisture', 'temperature', 'humidity', 'pH', 'light_intensity']
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df[features])

# --- Time-Delay Setup (TDANN: lookback window) ---
def create_sequences(data, lookback):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i])
        y.append(data[i, 0])  # Predict soil_moisture as proxy for growth
    return np.array(X), np.array(y)

lookback = 5
X, y = create_sequences(data_scaled, lookback)

# --- TDANN Model (LSTM-based) ---
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(32))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
history = model.fit(X, y, epochs=30, batch_size=16, validation_split=0.2, shuffle=False)

# --- Plot Loss ---
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('TDANN Training Loss')
plt.legend()
plt.show()

# --- Save Model ---
model.save("tdann_pnsm_model.h5")
