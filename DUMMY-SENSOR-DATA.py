import firebase_admin
from firebase_admin import credentials, db
import datetime
import random
import time
import os
import base64
import tempfile # For creating temporary files from environment variables

def run(): # Added the run function
    # --- Firebase Secure Setup (Render-Compatible) ---
    firebase_cred_path = None # Initialize to None
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
            # Ensure 'agriastrax-website-firebase-adminsdk-fbsvc-36cdff39c2.json' is in your project root
            cred = credentials.Certificate("agriastrax-website-firebase-adminsdk-fbsvc-36cdff39c2.json")
            print("Firebase credentials loaded from local file.")

        # Prevent double initialization in case this script is run multiple times
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred, {
                'databaseURL': 'https://agriastrax-website-default-rtdb.firebaseio.com/'
            })
        print("Firebase initialized successfully for data inserter.")
    except Exception as e:
        print(f"❌ Error initializing Firebase for insert-sample-data: {e}")
        # In a thread, exiting might abruptly stop the main app, consider logging and returning
        return # Return from run function if Firebase fails to initialize
    finally:
        # Clean up the temporary file if created from environment variable
        if firebase_key_b64 and firebase_cred_path and os.path.exists(firebase_cred_path):
            os.remove(firebase_cred_path)
            print(f"Cleaned up temporary Firebase cred file: {firebase_cred_path}")

    ref = db.reference('sensors/farm1')

    # Define possible crop stages
    CROP_STAGES = ['seed', 'sprout', 'vegetative', 'flowering', 'mature']

    # Insert 10 past samples
    print("Inserting 10 enhanced dummy sensor readings with all 12 features (including crop_stage and growth_factor)...")
    for i in range(10):
        sample_data = {
            'timestamp': (datetime.datetime.now() - datetime.timedelta(minutes=(10 - i)*5)).isoformat(),
            'soil_moisture': round(random.uniform(20, 80), 2),
            'temperature': round(random.uniform(25, 40), 2),
            'humidity': round(random.uniform(40, 90), 2),
            'pH': round(random.uniform(5.5, 7.5), 2), # Corrected to 'pH' for consistency with dashboard
            'light_intensity': random.randint(300, 800),
            'N': random.randint(0, 120),
            'P': random.randint(0, 60),
            'K': random.randint(0, 200),
            'rainfall': round(random.uniform(0, 200), 2),
            'crop_stage': random.choice(CROP_STAGES),
            'growth_factor': round(random.uniform(0.3, 1.0), 2)  # simulating energy/photosynthesis score
        }
        try:
            ref.push(sample_data)
        except Exception as e:
            print(f"❌ Error pushing initial sample data to Firebase: {e}")
            break # Stop inserting if there's an error
    print("Successfully inserted 10 enhanced dummy sensor readings.")

    # Simulate live updates every 10s
    print("\nSimulating live updates. New data will be inserted every 10 seconds. Press Ctrl+C to stop.")
    while True:
        current_timestamp = datetime.datetime.now().isoformat()
        
        # Introduce some realistic variations
        current_soil_moisture = round(random.uniform(20, 60), 2)
        current_temperature = round(random.uniform(28, 35), 2)
        current_humidity = round(random.uniform(50, 80), 2)
        current_pH = round(random.uniform(6.0, 7.2), 2)
        current_light_intensity = random.randint(400, 700)
        current_N = random.randint(20, 100)
        current_P = random.randint(10, 50)
        current_K = random.randint(30, 150)
        current_rainfall = round(random.uniform(0, 120), 2)
        
        # Simulate a progression in crop stage over time (very basic simulation)
        # You might want a more sophisticated logic here based on actual time or growth_factor
        # For now, it will pick a random stage, but you could make it sequential.
        current_crop_stage = random.choice(CROP_STAGES) 
        
        # Growth factor could be influenced by light, temperature, moisture
        # A simple example: higher light, temp, and moisture might lead to higher growth_factor
        current_growth_factor = round(
            random.uniform(0.3, 1.0) * (current_light_intensity / 800) * (current_temperature / 40) * (current_soil_moisture / 80), 2
        )
        # Clamp growth_factor to [0.3, 1.0] if the above calculation goes outside.
        current_growth_factor = max(0.3, min(1.0, current_growth_factor))

        try:
            ref.push({
                'timestamp': current_timestamp,
                'soil_moisture': current_soil_moisture,
                'temperature': current_temperature,
                'humidity': current_humidity,
                'pH': current_pH, # Corrected to 'pH' for consistency with dashboard
                'light_intensity': current_light_intensity,
                'N': current_N,
                'P': current_P,
                'K': current_K,
                'rainfall': current_rainfall,
                'crop_stage': current_crop_stage,
                'growth_factor': current_growth_factor
            })
            print(f"Inserted real-time enriched sensor data at {current_timestamp}. Data: {current_crop_stage}, {current_growth_factor}")
        except Exception as e:
            print(f"❌ Error pushing real-time data to Firebase: {e}")
            # If continuous errors occur, you might want a more robust error handling
            # e.g., exponential backoff, or stop trying to push data for a while.
        time.sleep(10)

if __name__ == "__main__":
    run() # Call the run function when the script is executed directly