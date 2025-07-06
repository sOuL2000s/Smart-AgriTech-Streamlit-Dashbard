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
    # Fallback for local development if environment variable is not set
    # Ensure 'agriastrax-website-firebase-adminsdk-fbsvc-36cdff39c2.json' is in your project root
    try:
        cred = credentials.Certificate("agriastrax-website-firebase-adminsdk-fbsvc-36cdff39c2.json")
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred, {
                'databaseURL': 'https://agriastrax-website-default-rtdb.firebaseio.com/'
            })
        print("Firebase initialized using local credentials file.")
    except Exception as e:
        raise ValueError(f"❌ Environment variable 'FIREBASE_KEY_B64' not set and local credentials file not found or invalid: {e}")
else:
    # Decode and write to temp file for environment variable
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
    # Clean up the temporary file after initialization
    os.remove(key_path)
    print("Firebase initialized using environment variable.")


# --- Simulate or Read Sensor Data ---
def read_sensor_data():
    """
    Generates simulated sensor data, including soil temperature, soil moisture,
    and a 'growth_trigger' based on these two parameters.
    """
    soil_temp = round(random.uniform(20, 35), 2)
    soil_moisture = round(random.uniform(20, 80), 2)
    
    # Simulate plant decision to grow based on soil temp + moisture
    # growth_trigger is 1 if soil_temp > 22 and soil_moisture > 30, otherwise 0
    growth_trigger = 1 if soil_temp > 22 and soil_moisture > 30 else 0

    return {
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'soil_moisture': soil_moisture,
        'temperature': soil_temp, # Renamed to 'temperature' for consistency with dashboard
        'humidity': round(random.uniform(30, 80), 2),
        'pH': round(random.uniform(5.5, 7.5), 2),
        'light_intensity': round(random.uniform(100, 1000), 2),
        'growth_trigger': growth_trigger # New field added
    }

# --- Push to Firebase ---
def push_to_firebase(sensor_data):
    """Pushes the given sensor data to the Firebase Realtime Database."""
    ref = db.reference('sensors/farm1')
    ref.push(sensor_data)
    print(f"Uploaded: {sensor_data}")

# --- Main Loop ---
if __name__ == '__main__':
    print("Starting live sensor data simulation. Data will be uploaded to Firebase every 10 seconds.")
    print("Press Ctrl+C to stop the script.")
    while True:
        data = read_sensor_data()
        push_to_firebase(data)
        time.sleep(10)  # 10-second interval