# Phase 2: EDA and Preprocessing (Firebase to Pandas, Secure Version)

import pandas as pd
import firebase_admin
from firebase_admin import credentials, db
import matplotlib.pyplot as plt
import seaborn as sns
import os
import base64
import tempfile

# --- Load Firebase credentials securely from base64 env variable ---
firebase_key_b64 = os.getenv("FIREBASE_KEY_B64")
if not firebase_key_b64:
    raise ValueError("❌ FIREBASE_KEY_B64 environment variable not set.")

decoded = base64.b64decode(firebase_key_b64).decode("utf-8")
with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w") as f:
    f.write(decoded)
    key_path = f.name

# --- Firebase Setup ---
if not firebase_admin._apps:
    cred = credentials.Certificate(key_path)
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://agriastrax-website-default-rtdb.firebaseio.com/'
    })

# --- Fetch Data ---
def fetch_data():
    ref = db.reference('sensors/farm1')
    snapshot = ref.get()
    if not snapshot:
        print("❌ No data found")
        return pd.DataFrame()
    return pd.DataFrame(snapshot).T

# --- Preprocessing ---
def preprocess(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    df = df.apply(pd.to_numeric, errors='ignore')
    df = df.reset_index(drop=True)
    return df

# --- EDA Plotting ---
def plot_eda(df):
    sns.set(style="darkgrid")
    plt.figure(figsize=(12, 6))
    for col in ['soil_moisture', 'temperature', 'humidity', 'pH', 'light_intensity']:
        if col in df.columns:
            plt.plot(df['timestamp'], df[col], label=col)
    plt.legend()
    plt.title('IoT Sensor Time Series')
    plt.xlabel('Timestamp')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# --- Run the Script ---
df_raw = fetch_data()
if not df_raw.empty:
    df_clean = preprocess(df_raw)
    print(df_clean.head())
    plot_eda(df_clean)
else:
    print("No sensor data available.")
