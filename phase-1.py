import time
import random
import firebase_admin
from firebase_admin import credentials, db
import os
import base64
import tempfile

# --- Load Firebase Key from Environment ---
firebase_key_b64 = os.getenv("FIREBASE_KEY_B64")
if not firebase_key_b64:
    raise ValueError("❌ Environment variable 'FIREBASE_KEY_B64' not set.")

# Decode and write to temp file
firebase_key_json = base64.b64decode(firebase_key_b64).decode('utf-8')
with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w") as f:
    f.write(firebase_key_json)
    key_path = f.name

# --- Initialize Firebase ---
if not firebase_admin._apps:
    cred = credentials.Certificate(key_path)
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://agriastrax-website-default-rtdb.firebaseio.com/'
    })

# --- Simulate or Read Sensor Data ---
def read_sensor_data():
    return {
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'soil_moisture': round(random.uniform(20, 80), 2),
        'temperature': round(random.uniform(22, 35), 2),
        'humidity': round(random.uniform(30, 80), 2),
        'pH': round(random.uniform(5.5, 7.5), 2),
        'light_intensity': round(random.uniform(100, 1000), 2)
    }

# --- Push to Firebase ---
def push_to_firebase(sensor_data):
    ref = db.reference('sensors/farm1')
    ref.push(sensor_data)
    print(f"Uploaded: {sensor_data}")

# --- Main Loop ---
if __name__ == '__main__':
    while True:
        data = read_sensor_data()
        push_to_firebase(data)
        time.sleep(10)  # 10-second interval
