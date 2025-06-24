import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.layers import Input
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load cleaned data
df = pd.read_csv("cleaned_sensor_data.csv")

# If timestamp is missing, generate synthetic ones
df['timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='h')
df.set_index('timestamp', inplace=True)
print("Columns in CSV:", df.columns.tolist())


# Features used for prediction
features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df[features])

# Create sequences for LSTM (lookback window = 5)
def create_sequences(data, lookback=5):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        y.append(data[i, 3])  # Predicting temperature (index 3), or choose another
    return np.array(X), np.array(y)

X, y = create_sequences(data_scaled)

# TDANN (LSTM-based) Model
model = Sequential()
model.add(Input(shape=(X.shape[1], X.shape[2])))
model.add(LSTM(64, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(32))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(X, y, epochs=30, batch_size=16, validation_split=0.2)

# Save the model
model.save("tdann_pnsm_model.keras")
print("Model saved as tdann_pnsm_model.keras")

# Optional: Plot training loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("TDANN Training Loss")
plt.legend()
plt.show()
