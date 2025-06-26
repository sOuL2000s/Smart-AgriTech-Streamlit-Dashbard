import firebase_admin
from firebase_admin import credentials, db
import datetime
import random
import time
import os
import base64
import tempfile

# --- Firebase Init (Render-Compatible) ---
firebase_key_b64 = os.getenv("FIREBASE_KEY_B64")

try:
    if firebase_key_b64:
        decoded_json = base64.b64decode(firebase_key_b64).decode('utf-8')
        with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.json') as f:
            f.write(decoded_json)
            firebase_cred_path = f.name
        cred = credentials.Certificate(firebase_cred_path)
    else:
        raise ValueError("FIREBASE_KEY_B64 environment variable not found")

    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://agriastrax-website-default-rtdb.firebaseio.com/'
        })
except Exception as e:
    print(f"❌ Firebase initialization failed: {e}")
    exit()

# --- Reference Path ---
ref = db.reference('sensors/farm1')

# --- Insert 10 Past Samples ---
for i in range(10):
    sample_data = {
        'timestamp': (datetime.datetime.now() - datetime.timedelta(minutes=(10 - i) * 5)).isoformat(),
        'soil_moisture': round(random.uniform(20, 80), 2),
        'temperature': round(random.uniform(25, 40), 2),
        'humidity': round(random.uniform(40, 90), 2),
        'pH': round(random.uniform(5.5, 7.5), 2),
        'light_intensity': random.randint(300, 800),
        'N': random.randint(0, 120),
        'P': random.randint(0, 60),
        'K': random.randint(0, 200),
        'ph': round(random.uniform(5.0, 7.5), 2),
        'rainfall': round(random.uniform(0, 200), 2)
    }
    ref.push(sample_data)

print("Inserted 10 dummy sensor readings.")

# --- Simulate Real-Time Updates Every 10s ---
while True:
    try:
        ref.push({
            'timestamp': datetime.datetime.now().isoformat(),
            'soil_moisture': random.uniform(20, 60),
            'temperature': random.uniform(28, 35),
            'humidity': random.uniform(50, 80),
            'pH': round(random.uniform(6.0, 7.2), 2),
            'light_intensity': random.randint(400, 700),
            'N': random.randint(20, 100),
            'P': random.randint(10, 50),
            'K': random.randint(30, 150),
            'ph': round(random.uniform(5.8, 7.2), 2),
            'rainfall': round(random.uniform(0, 120), 2)
        })
        print("📡 Inserted real-time sensor data.")
        time.sleep(10)
    except KeyboardInterrupt:
        print("Stopped data simulation.")
        break
    except Exception as e:
        print(f"Error during data push: {e}")
        time.sleep(5)
