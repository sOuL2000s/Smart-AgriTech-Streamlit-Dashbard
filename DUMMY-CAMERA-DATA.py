# dummy_camera_simulator.py

import time
import random
import json
from datetime import datetime
import os
import base64
import tempfile

# Import Firebase Admin SDK
import firebase_admin
from firebase_admin import credentials, db

# Simulated growth stages
growth_stages = ["Germination", "Vegetative", "Flowering", "Maturity", "Wilting", "Yellowing"]

# --- Firebase Secure Setup (Render-Compatible) ---
# This part is adapted from your main dashboard script to make it runnable independently.
# It tries to get the key from an environment variable first, then from a local file.
firebase_cred_path = None
try:
    firebase_key_b64 = os.getenv("FIREBASE_KEY_B64")
    if firebase_key_b64:
        decoded_json = base64.b64decode(firebase_key_b64).decode('utf-8')
        with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.json') as f:
            f.write(decoded_json)
        firebase_cred_path = f.name
        cred = credentials.Certificate(firebase_cred_path)
        print("Firebase credentials loaded from environment variable.")
    else:
        # Fallback for local development if environment variable is not set
        cred = credentials.Certificate("agriastrax-website-firebase-adminsdk-fbsvc-36cdff39c2.json")
        print("Firebase credentials loaded from local file.")

    # Prevent double initialization in case this script is run multiple times or in certain environments
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://agriastrax-website-default-rtdb.firebaseio.com/'
        })
    print("Firebase initialized successfully for camera simulator.")
except Exception as e:
    print(f"❌ Error initializing Firebase for dummy_camera_simulator: {e}")
    # It's crucial to handle this error. If Firebase can't init, the script can't push data.
    # For a simulator, you might just let it run and log errors, but it won't push data.
    firebase_admin = None # Mark Firebase as not initialized

finally:
    # Clean up the temporary file if created from environment variable
    if firebase_key_b64 and firebase_cred_path and os.path.exists(firebase_cred_path):
        os.remove(firebase_cred_path)
        print(f"Cleaned up temporary Firebase cred file: {firebase_cred_path}")


def generate_dummy_growth_event():
    event = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "stage": random.choice(growth_stages),
        "alert": random.choice([
            "Healthy Growth",
            "Low Leaf Color Index",
            "Possible Disease Detected",
            "Needs Fertilizer",
            "Check Irrigation"
        ])
    }
    return event

def run():
    print("Starting dummy camera feed simulation...")

    if firebase_admin is None:
        print("Firebase is not initialized. Cannot push data to Firebase. Simulation will run but only print locally.")
        # Fallback to local file if Firebase fails, or just print
        local_file_only = True
        # OUTPUT_FILE = "simulated_growth_data.json" # Re-enable if you want local file fallback
    else:
        local_file_only = False
        try:
            # Get a database reference to store camera data
            camera_ref = db.reference('camera_feed/farm1')
            print("Connected to Firebase path: camera_feed/farm1")
        except Exception as e:
            print(f"❌ Error getting Firebase database reference: {e}. Falling back to local file.")
            local_file_only = True


    while True:
        event = generate_dummy_growth_event()
        
        if not local_file_only:
            try:
                # Push data to Firebase
                camera_ref.push(event)
                print(f"Simulated Data pushed to Firebase: {event}")
            except Exception as e:
                print(f"❌ Error pushing data to Firebase: {e}. Falling back to local printing.")
                # If a persistent Firebase error, just print locally
                local_file_only = True # Stop trying to push to Firebase
                print("Simulated Data (local print):", event)
        else:
            # If Firebase is not initialized or failed, print locally
            print("Simulated Data (local print):", event)
            # You can re-enable writing to a local JSON file here if desired as a fallback:
            # with open(OUTPUT_FILE, "w") as f:
            #     json.dump(event, f)

        time.sleep(10)  # Generate every 10 seconds

if __name__ == "__main__":
    run()