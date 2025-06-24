# Add this version in place of your current insert-sample-data.py
import firebase_admin
from firebase_admin import credentials, db
import datetime
import random
import time

# Firebase Init
cred = credentials.Certificate("agriastrax-website-firebase-adminsdk-fbsvc-36cdff39c2.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://agriastrax-website-default-rtdb.firebaseio.com/'
})

ref = db.reference('sensors/farm1')

# Insert 10 past samples
for i in range(10):
    sample_data = {
        'timestamp': (datetime.datetime.now() - datetime.timedelta(minutes=(10 - i)*5)).isoformat(),
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

print("Inserted 10 enhanced dummy sensor readings with all 10 features.")

# Simulate live updates every 10s
while True:
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
    print("Inserted real-time enriched sensor data.")
    time.sleep(10)
