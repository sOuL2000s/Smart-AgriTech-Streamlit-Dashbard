import time
import random
import json
from datetime import datetime, timedelta
import os
import base64
import tempfile
import threading
import io
import functools # For caching

from flask import Flask, jsonify, request, send_file, Response
from flask_cors import CORS

import firebase_admin
from firebase_admin import credentials, db

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
import joblib
from gtts import gTTS
import requests # Keep requests for LibreTranslate API calls

app = Flask(__name__)
CORS(app)

# --- Global Variables for Firebase, Models, Scalers, etc. ---
firebase_app = None
model = None # For growth prediction
input_scaler = None # For growth prediction model input
output_scaler = None # For growth prediction model output
crop_encoder = None # For encoding crop types for growth prediction
market_price_model = None
market_crop_encoder = None # For encoding crop types for market price prediction
market_price_features = None # List of features expected by the market price model
all_crop_labels = [] # List of all known crop labels from training data
firebase_db_ref = None  # Global reference for Firebase DB (sensor data)
firebase_camera_ref = None  # Global reference for Firebase camera feed

# --- New Global Variable for Simulation Mode ---
simulation_mode = False

# --- LibreTranslate Configuration ---
# IMPORTANT: Replace with your actual LibreTranslate instance URL if not using the public one.
# For production, it's highly recommended to host your own LibreTranslate instance or use a more robust translation service.
LIBRE_TRANSLATE_BASE_URL = os.getenv("LIBRE_TRANSLATE_URL", "https://libretranslate.com")

# --- Dynamic Translation Function with Caching ---
@functools.lru_cache(maxsize=128) # Cache up to 128 unique translations
def translate_dynamic(text, target_lang='en', source_lang='en'):
    """
    Translates text using LibreTranslate API.
    Uses caching to avoid repeated API calls for the same text.
    """
    if target_lang == source_lang:
        return text # No translation needed if source and target are the same

    payload = {
        'q': text,
        'source': source_lang,
        'target': target_lang,
        'format': 'text'
    }
    try:
        response = requests.post(f"{LIBRE_TRANSLATE_BASE_URL}/translate", json=payload)
        response.raise_for_status() # Raise an exception for HTTP errors
        return response.json()['translatedText']
    except requests.exceptions.RequestException as e:
        print(f"Translation error for '{text}' to '{target_lang}': {e}")
        return text # Fallback to original text on error

# --- Helper function for formatting and translating dynamic messages ---
def format_and_translate(key_template, lang, variables={}, source_lang='en'):
    """
    Formats a message template with variables and then translates it.
    """
    # Define base English messages for dynamic formatting.
    # These are the *source* messages that will be translated.
    base_messages = {
        'no_data': "No sensor data available to provide advice.",
        'npk_low': "🌱 **{nutrient} is low ({value:.1f})**: {message}",
        'npk_high': "🌱 **{nutrient} is high ({value:.1f})**: {message}",
        'soil_moisture_low': "💧 **Soil Moisture is low ({sm:.1f}%)**: {message}",
        'soil_moisture_high': "💧 **Soil Moisture is high ({sm:.1f}%)**: {message}",
        'temp_low': "🌡️ **Temperature is low ({temp:.1f}°C)**: {message}",
        'temp_high': "🌡️ **Temperature is high ({temp:.1f}°C)**: {message}",
        'humidity_low': "💨 **Humidity is low ({hum:.1f}%)**: {message}",
        'humidity_high': "💨 **Humidity is high ({hum:.1f}%)**: {message}",
        'ph_low': "🧪 **pH is low ({ph_val:.1f})**: {message}",
        'ph_high': "🧪 **pH is high ({ph_val:.1f})**: {message}",
        'ph_off': "🧪 **pH is off ({ph_val:.1f})**: {message}",
        'light_low': "☀️ **Light Intensity is low ({light:.1f} lux)**: {message}",
        'light_high': "☀️ **Light Intensity is high ({light:.1f} lux)**: {message}",
        'rainfall_low_msg': "🌧️ **Rainfall is low ({rain:.1f} mm)**: {message}",
        'rainfall_high_msg': "🌧️ **Rainfall is high ({rain:.1f} mm)**: {message}",
        'all_good': "✅ All major parameters look good! Keep monitoring regularly for optimal growth.",
        'npk_n_low': "Consider applying nitrogen-rich fertilizer.",
        'npk_n_high': "Excess nitrogen can promote leafy growth over fruit/flower development.",
        'npk_p_low': "Consider applying phosphorus fertilizer for root development.",
        'npk_p_high': "High phosphorus can lock up other nutrients.",
        'npk_k_low': "Consider applying potassium fertilizer for overall plant health and fruit quality.",
        'npk_k_high': "Excess potassium can interfere with calcium and magnesium uptake.",
        'wheat_sm_low': "Irrigate lightly – wheat needs 35–50% soil moisture.",
        'rice_sm_low': "Rice needs high moisture. Ensure proper irrigation.",
        'maize_sm_low': "Maize needs moderate soil moisture levels.",
        'banana_sm_low': "Keep soil consistently moist for banana.",
        'mango_sm_high': "Avoid waterlogging. Mango needs well-drained soil.",
        'grapes_sm_high': "Grapes prefer drier soil – avoid overwatering.",
        'cotton_sm_low': "Cotton requires moderate moisture during flowering.",
        'millet_sorghum_sm_low': "These are drought-resistant crops but still need minimal moisture.",
        'jute_sm_low': "Jute requires ample moisture during growth.",
        'pomegranate_sm_high': "Avoid overwatering pomegranate.",
        'melon_sm_low': "Melons need consistent watering, especially during fruiting.",
        'coconut_sm_low': "Coconut palms need high moisture levels.",
        'mothbeans_sm_low': "Mothbeans are drought-tolerant but need minimal irrigation during flowering.",
        'mungbean_sm_low': "Ensure regular irrigation during flowering and pod formation.",
        'blackgram_sm_low': "Maintain moderate moisture especially during flowering.",
        'lentil_sm_low': "Lentils need low to moderate moisture.",
        'general_sm_low': "General advice: Consider irrigation to prevent drought stress.",
        'general_sm_high': "General advice: Ensure good drainage to prevent waterlogging.",
        'wheat_temp_high': "Provide shade or irrigate in evening – temp is too high for wheat.",
        'rice_temp_high': "Too hot for rice. Consider evening irrigation or shade.",
        'maize_temp_low': "Maize prefers warm weather (20–30°C).",
        'banana_temp_low': "Banana is sensitive to cold – ensure warm conditions.",
        'mango_temp_low': "Mango requires warmer temperatures (>20°C).",
        'cotton_temp_low': "Cotton thrives in warm temperatures.",
        'millet_sorghum_temp_low': "Warm climate is ideal for millet/sorghum.",
        'coffee_temp_low': "Coffee thrives in 18–24°C range.",
        'jute_temp_low': "Jute grows well in 25–30°C.",
        'papaya_temp_low': "Papaya prefers 21–33°C range.",
        'pomegranate_temp_low': "Ideal temperature is above 20°C.",
        'melon_temp_low': "Ensure temperature is warm (>25°C).",
        'coconut_temp_low': "Ideal temperature for coconut is above 25°C.",
        'mothbeans_temp_low': "Temperature should be above 22°C.",
        'mungbean_temp_low': "Mungbean requires warm conditions for optimal growth.",
        'blackgram_temp_low': "Ideal temperature range is 25–35°C.",
        'lentil_temp_low': "Lentils grow well in 18–30°C.",
        'general_temp_low': "General advice: Cold temperatures can stunt growth. Consider protective measures.",
        'general_temp_high': "General advice: High temperatures can cause heat stress. Ensure adequate water and shade.",
        'wheat_hum_high': "Watch out for fungal infections – ensure airflow.",
        'rice_hum_low': "Increase ambient humidity or use mulch.",
        'banana_hum_low': "Banana requires high humidity. Consider misting or mulching.",
        'grapes_hum_high': "High humidity may lead to fungal infections.",
        'coffee_hum_low': "Coffee prefers high humidity.",
        'orange_hum_high': "Prune trees to improve airflow and prevent fungal issues.",
        'general_hum_low': "General advice: Low humidity can cause wilting. Consider misting or increasing soil moisture.",
        'general_hum_high': "General advice: High humidity increases risk of fungal diseases. Ensure good ventilation.",
        'wheat_ph_low': "Slightly acidic – consider applying lime to raise pH.",
        'rice_ph_off': "Maintain slightly acidic soil for rice (pH 5.5–6.5).",
        'maize_ph_off': "Maintain soil pH between 5.8–7.0.",
        'papaya_ph_low': "Slightly acidic to neutral soil is best for papaya.",
        'orange_ph_off': "Ideal soil pH for orange is 6.0–7.5.",
        'general_ph_very_low': "General advice: Soil is too acidic. Apply lime to increase pH and improve nutrient availability.",
        'general_ph_very_high': "General advice: Soil is too alkaline. Apply sulfur or organic matter to decrease pH.",
        'general_ph_off': "General advice: Optimal pH range for most crops is 5.5-7.5. Adjust as needed.",
        'wheat_light_low': "Ensure the crop gets enough sunlight.",
        'rice_light_low': "Ensure rice gets full sun exposure.",
        'general_light_low': "General advice: Insufficient light can hinder photosynthesis. Consider supplemental lighting or pruning.",
        'general_light_high': "General advice: Excessive light can cause scorching. Consider shading during peak hours.",
        'intro': "Based on current conditions, you might consider: ",
        'outro': ". Please consult local agricultural experts for precise recommendations.",
        'acid_tolerant': "acid-tolerant crops like blueberries, potatoes, or specific rice varieties",
        'alkaline_tolerant': "alkaline-tolerant crops such as asparagus, spinach, or specific varieties of alfalfa",
        'neutral_ph': "a wide range of crops thrive in neutral to slightly acidic pH (5.5-7.5), including wheat, maize, and most vegetables",
        'heat_tolerant': "heat-tolerant crops like millet, sorghum, cotton, or some varieties of beans",
        'cold_hardy': "cold-hardy crops such as wheat (winter varieties), barley, oats, or peas",
        'warm_season': "warm-season crops like maize, rice (tropical), most vegetables, and fruits",
        'drought_resistant': "drought-resistant crops like millet, sorghum, chickpeas, or certain types of beans (e.g., mothbeans)",
        'water_loving': "water-loving crops such as rice, sugarcane, jute, or crops that tolerate temporary waterlogging",
        'moderate_rainfall': "crops suitable for moderate rainfall, including wheat, maize, and many vegetables",
        'very_dry': "very drought-tolerant crops (e.g., desert-adapted melons or some herbs)",
        'very_wet': "semi-aquatic crops or those highly tolerant to waterlogging (e.g., taro, some rice varieties if poorly drained)",
        'no_specific': "No specific recommendations, as current conditions are unusual or general.",
        'connected': 'Connected',
        'disconnected': 'Disconnected',
        'growth_stage': 'Growth Stage',
        'advisory': 'Advisory',
        'timestamp': 'Timestamp',
        'last_frame': 'Last Frame',
        'no_camera_data': 'No Camera Data'
    }

    template = base_messages.get(key_template, key_template) # Fallback to key_template if not found
    formatted_text = template.format(**variables)
    return translate_dynamic(formatted_text, target_lang, source_lang)


# Simulated growth stages and crop stages
growth_stages = ["Germination", "Vegetative", "Flowering", "Maturity", "Wilting", "Yellowing"]
CROP_STAGES = ['seed', 'sprout', 'vegetative', 'flowering', 'mature']

# --- Initialization Functions ---
def initialize_firebase():
    """Initializes Firebase Admin SDK."""
    global firebase_app, firebase_db_ref, firebase_camera_ref
    if firebase_admin._apps: # Prevent double initialization
        firebase_app = firebase_admin.get_app()
        print("Firebase already initialized.")
        return

    firebase_key_b64 = os.getenv("FIREBASE_KEY_B64")
    firebase_cred_path = None
    try:
        if firebase_key_b64:
            decoded_json = base64.b64decode(firebase_key_b64).decode('utf-8')
            with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.json') as f:
                f.write(decoded_json)
            firebase_cred_path = f.name
            cred = credentials.Certificate(firebase_cred_path)
            print("Firebase credentials loaded from environment variable.")
        else:
            cred = credentials.Certificate("agriastrax-website-firebase-adminsdk-fbsvc-36cdff39c2.json")
            print("Firebase credentials loaded from local file (development fallback).")

        firebase_app = firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://agriastrax-website-default-rtdb.firebaseio.com/' # Replace with your actual Firebase DB URL
        })
        firebase_db_ref = db.reference('sensors/farm1') # Path for sensor data
        firebase_camera_ref = db.reference('camera_feed/farm1') # Path for camera feed
        print("Firebase initialized successfully and references obtained.")

    except Exception as e:
        print(f"❌ Firebase initialization failed: {e}")
        firebase_app = None
        firebase_db_ref = None
        firebase_camera_ref = None
    finally:
        if firebase_key_b64 and firebase_cred_path and os.path.exists(firebase_cred_path):
            try:
                os.remove(firebase_cred_path)
                print(f"Cleaned up temporary Firebase cred file: {firebase_cred_path}")
            except Exception as e_clean:
                print(f"Warning: Could not delete temporary Firebase cred file {firebase_cred_path}: {e_clean}")

def load_models_and_scalers():
    """Loads AI models, scalers, and crop encoders."""
    global model, input_scaler, output_scaler, crop_encoder, \
           market_price_model, market_crop_encoder, market_price_features, all_crop_labels

    # Load Crop Labels from CSV (for encoder fitting)
    try:
        crop_df_for_labels = pd.read_csv("cleaned_sensor_data.csv")
        all_crop_labels = sorted(crop_df_for_labels['label'].unique().tolist())
        crop_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        crop_encoder.fit(np.array(all_crop_labels).reshape(-1, 1))
        market_crop_encoder = crop_encoder # Use the same encoder for market price
        print(f"Crop labels loaded: {len(all_crop_labels)} unique crops found.")
    except FileNotFoundError:
        print("❌ 'cleaned_sensor_data.csv' not found. Crop labels and encoder will be limited or empty.")
        all_crop_labels = ["Wheat", "Rice", "Maize", "Banana", "Mango", "Grapes", "Cotton", "Millet", "Sorghum", "Coffee", "Jute", "Pomegranate", "Melon", "Coconut", "Mothbeans", "Mungbean", "Blackgram", "Lentil"] # Default dummy crops
        crop_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        # Fit with dummy labels if CSV not found, so encoder is usable
        if all_crop_labels:
            crop_encoder.fit(np.array(all_crop_labels).reshape(-1, 1))
        market_crop_encoder = crop_encoder
    except Exception as e:
        print(f"❌ Error loading 'cleaned_sensor_data.csv': {e}")
        all_crop_labels = ["Wheat", "Rice", "Maize", "Banana", "Mango", "Grapes", "Cotton", "Millet", "Sorghum", "Coffee", "Jute", "Pomegranate", "Melon", "Coconut", "Mothbeans", "Mungbean", "Blackgram", "Lentil"] # Default dummy crops
        crop_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        if all_crop_labels:
            crop_encoder.fit(np.array(all_crop_labels).reshape(-1, 1))
        market_crop_encoder = crop_encoder

    # Load AI Model
    try:
        model = tf.keras.models.load_model("tdann_pnsm_model.keras") # Or 'models/growth_prediction_model.h5'
        print("AI model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading AI model: {e}")
        model = None

    # Load Scalers
    try:
        input_scaler = joblib.load('tdann_input_scaler.joblib') # Or 'models/input_scaler.pkl'
        output_scaler = joblib.load('tdann_output_scaler.joblib') # Or 'models/output_scaler.pkl'
        print("Input and Output scalers loaded successfully.")
    except FileNotFoundError:
        print("❌ Scaler files not found. Using newly initialized scalers. Predictions may be inaccurate.")
        input_scaler = MinMaxScaler()
        output_scaler = MinMaxScaler()
    except Exception as e:
        print(f"❌ Error loading scalers: {e}")
        input_scaler = MinMaxScaler()
        output_scaler = MinMaxScaler()

    # Market Price Predictor Setup (Simulated Training if no pre-trained model)
    try:
        market_price_model = joblib.load('market_price_model.joblib') # Or 'models/market_price_model.pkl'
        print("Market price prediction model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading market price model: {e}. Training a dummy model.")
        def generate_market_price_data(num_samples=1000):
            data = []
            crops = all_crop_labels if all_crop_labels else ['wheat', 'rice', 'maize']
            for _ in range(num_samples):
                crop_type = random.choice(crops)
                N = random.uniform(50, 150)
                P = random.uniform(20, 60)
                K = random.uniform(50, 200)
                temperature = random.uniform(20, 35)
                humidity = random.uniform(30, 80)
                soil_moisture = random.uniform(30, 70)
                light_intensity = random.uniform(300, 800)
                rainfall = random.uniform(0, 100)
                ph = random.uniform(5.5, 7.5)
                ds18b20_temperature = random.uniform(15, 40) # New sensor data

                base_price = 100
                if crop_type == 'wheat': price = base_price * 1.2
                elif crop_type == 'rice': price = base_price * 1.5
                elif crop_type == 'maize': price = base_price * 1.1
                else: price = base_price * 1.0
                price += (N / 10) + (P / 5) + (K / 10)
                price += (temperature - 25) * 2
                price += (humidity - 50) * 1.5
                price += (soil_moisture - 50) * 0.5
                price += (light_intensity - 500) * 0.1
                price += (rainfall - 50) * 0.2
                price += (ph - 6.5) * 5
                price += (ds18b20_temperature - 25) * 1.0 # Factor in new temperature sensor
                price += random.uniform(-10, 10)
                price = max(50, price)
                data.append([N, P, K, temperature, humidity, soil_moisture, light_intensity, rainfall, ph, ds18b20_temperature, crop_type, price])
            df_prices = pd.DataFrame(data, columns=['N', 'P', 'K', 'temperature', 'humidity', 'soil_moisture', 'light_intensity', 'rainfall', 'ph', 'ds18b20_temperature', 'crop_type', 'price'])
            return df_prices

        df_prices = generate_market_price_data(num_samples=2000)
        market_price_features = ['N', 'P', 'K', 'temperature', 'humidity', 'soil_moisture', 'light_intensity', 'rainfall', 'ph', 'ds18b20_temperature']

        X_numerical = df_prices[market_price_features]
        if market_crop_encoder and (not market_crop_encoder.categories_ or not market_crop_encoder.categories_[0].tolist()):
             market_crop_encoder.fit(np.array(df_prices['crop_type'].unique()).reshape(-1, 1))

        X_categorical = market_crop_encoder.transform(df_prices[['crop_type']])
        X_categorical_df = pd.DataFrame(X_categorical, columns=market_crop_encoder.get_feature_names_out(['crop_type']))

        X_train_market = pd.concat([X_numerical, X_categorical_df], axis=1)
        y_train_market = df_prices['price']

        market_price_model = LinearRegression()
        market_price_model.fit(X_train_market, y_train_market)
        print("Market price prediction model trained (simulated data).")


# --- Dummy Data Generation (for unavailable sensors and initial simulation) ---
def generate_dummy_sensor_data_values():
    """Generates dummy data for pH, NPK, Rainfall, DS18B20 Temperature, and other fields if not provided by ESP32."""
    return {
        "ph": round(random.uniform(5.5, 7.5), 2),
        "N": round(random.uniform(20, 100), 2),
        "P": round(random.uniform(10, 50), 2),
        "K": round(random.uniform(30, 120), 2),
        "rainfall": round(random.uniform(0, 10), 2),
        "ds18b20_temperature": round(random.uniform(15, 40), 2), # New DS18B20 temperature sensor
        "crop_stage": random.choice(CROP_STAGES),
        "growth_factor": round(random.uniform(0.5, 1.5), 2)
    }

def generate_dummy_camera_data(lang='en'):
    """Generates dummy camera data for demonstration, with translated advisories."""
    stage_options = ["Germination", "Vegetative", "Flowering", "Maturity", "Wilting", "Yellowing"]
    alert_options = ["Healthy Growth", "Low Leaf Color Index", "Possible Disease Detected", "Needs Fertilizer", "Check Irrigation"]

    translated_stage = format_and_translate('growth_stage', lang, {}) + ": " + translate_dynamic(random.choice(stage_options), lang)
    translated_alert = format_and_translate('advisory', lang, {}) + ": " + translate_dynamic(random.choice(alert_options), lang)

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "stage": translated_stage,
        "alert": translated_alert,
        "image_url": "https://placehold.co/150x150/E0E0E0/333333?text=Camera+Feed" # Placeholder image
    }

def generate_dummy_weather_data():
    """Generates simulated weather data."""
    now = datetime.now()
    current_temp = round(random.uniform(20, 30), 1)
    current_humidity = random.randint(50, 80)
    wind_speed = random.randint(5, 20)

    forecast = []
    for i in range(3): # Today, Tomorrow, Day after tomorrow
        day_offset = i
        day_name = (now + timedelta(days=day_offset)).strftime("%a")
        high_temp = round(random.uniform(current_temp + 2, current_temp + 8), 1)
        low_temp = round(random.uniform(current_temp - 8, current_temp - 2), 1)
        forecast.append({
            "day": "Today" if i == 0 else ("Tomorrow" if i == 1 else day_name),
            "high": high_temp,
            "low": low_temp
        })

    advisories = []
    if random.random() < 0.3:
        warnings = ["Heavy Rainfall expected tonight.", "High Wind Advisory issued.", "Frost warning for early morning.", "Heatwave alert."]
        advisories.append({"type": "warning", "message": random.choice(warnings)})
    if not advisories:
        advisories.append({"type": "info", "message": "No active advisories."})

    return {
        "current": {
            "temp": current_temp,
            "humidity": current_humidity,
            "wind": wind_speed,
            "timestamp": now.strftime("%H:%M")
        },
        "forecast": forecast,
        "advisories": advisories
    }

def generate_dummy_recent_events():
    """Generates simulated recent events."""
    events = [
        {"text": "Water detected in sector B", "time": "10:27 AM"},
        {"text": "Irrigation cycle completed", "time": "09:45 AM"},
        {"text": "Optimal light conditions detected", "time": "09:10 AM"},
        {"text": "Nutrient levels adjusted in field 1", "time": "Yesterday"},
        {"text": "Pest activity observed in north plot", "time": "2 days ago"},
        {"text": "Temperature spike recorded in greenhouse", "time": "3 days ago"},
    ]
    random.shuffle(events) # Shuffle to make it seem dynamic
    return events[:random.randint(3, 6)] # Return 3 to 6 random events

def generate_dummy_farm_health_data():
    """Generates simulated farm health index data."""
    overall_score = random.randint(60, 95)
    status = "Good"
    if overall_score < 75: status = "Fair"
    if overall_score < 60: status = "Poor"

    health_factors = {
        'soil-quality': random.randint(70, 98),
        'plant-health': random.randint(65, 95),
        'water-management': random.randint(60, 90),
        'pest-control': random.randint(60, 90),
        'environmental': random.randint(75, 98)
    }

    health_change = round(random.uniform(-5, 5), 0)

    return {
        "overallScore": overall_score,
        "status": status,
        "healthFactors": health_factors,
        "healthChange": health_change
    }

def generate_dummy_device_connectivity():
    """Generates simulated device connectivity data."""
    statuses = ['Online', 'Offline', 'Active', 'Inactive', 'Connected', 'Disconnected']

    camera_connected_count = random.randint(0, 3)
    soil_online_status = 'All Online' if random.random() > 0.1 else 'Some Offline'

    return {
        "gateway": random.choice(['Online', 'Offline']),
        "irrigation": random.choice(['Active', 'Inactive']),
        "camera": f"{camera_connected_count}/3 Connected",
        "soil": soil_online_status
    }

def generate_dummy_pest_scan_results():
    """Generates simulated pest and disease scan results."""
    results = [
        {"status": "No threats detected. Your crops are healthy!", "type": "success", "details": []},
        {"status": "Minor pest activity detected in Sector C. Recommend localized treatment.", "type": "warning", "details": ["Aphids (low concentration)"]},
        {"status": "Disease detected in Field 2. Immediate action required.", "type": "error", "details": ["Fungal blight (moderate)", "Leaf spot (minor)"]},
        {"status": "Potential nutrient deficiency in Sector A. Further analysis recommended.", "type": "info", "details": ["Nitrogen deficiency (early stage)"]},
    ]
    return random.choice(results)

def generate_dummy_resource_consumption():
    """Generates simulated daily resource consumption data."""
    return {
        "water_used": random.randint(1500, 4000), # Liters
        "energy_used": random.randint(30, 100), # kWh
        "nutrients_applied": random.randint(2, 10) # kg
    }

def generate_dummy_usage_history_data():
    """Generates simulated historical resource consumption data for a week."""
    history = []
    for i in range(7):
        date = (datetime.now() - timedelta(days=6-i)).strftime("%Y-%m-%d")
        history.append({
            "date": date,
            "water_used": random.randint(1000, 5000),
            "energy_used": random.randint(20, 120),
            "nutrients_applied": random.randint(1, 15)
        })
    return history


# --- Sensor Data Inserter and Camera Simulator Threads ---
def run_camera_simulator_thread():
    """Simulates camera feed data and pushes to Firebase."""
    print("Starting dummy camera feed simulation thread...")
    while True:
        if firebase_camera_ref:
            try:
                dummy_camera = generate_dummy_camera_data(lang='en') # Dummy data is always English, translation happens on fetch
                firebase_camera_ref.push(dummy_camera)
                snapshots = firebase_camera_ref.order_by_child('timestamp').get()
                if snapshots and len(snapshots) > 10:
                    oldest_keys = sorted(snapshots.keys(), key=lambda k: snapshots[k]['timestamp'])
                    for i in range(len(oldest_keys) - 10):
                        firebase_camera_ref.child(oldest_keys[i]).delete()
            except Exception as e:
                print(f"❌ Error pushing camera data to Firebase: {e}")
        time.sleep(10)

def run_sensor_data_inserter_thread():
    """Inserts initial dummy sensor data and then simulates live updates to Firebase."""
    global simulation_mode
    print("Starting sensor data inserter thread...")
    if firebase_db_ref is None:
        print("Firebase DB reference not initialized. Sensor data insertion will only print locally.")
        local_print_only = True
    else:
        local_print_only = False
        print("Connected to Firebase path for sensor data: sensors/farm1")

    print("Inserting 10 enhanced dummy sensor readings with all features (for initial data)...")
    for i in range(10):
        sample_data = {
            'timestamp': (datetime.now() - timedelta(minutes=(10 - i)*5)).isoformat(),
            'soil_moisture': round(random.uniform(20, 80), 2),
            'temperature': round(random.uniform(20, 40), 2),
            'humidity': round(random.uniform(30, 95), 2),
            'light_intensity': random.randint(200, 900),
            'ds18b20_temperature': round(random.uniform(15, 40), 2),
            **generate_dummy_sensor_data_values()
        }
        if not local_print_only:
            try:
                firebase_db_ref.push(sample_data)
            except Exception as e:
                print(f"❌ Error pushing initial sample data to Firebase: {e}. Falling back to local printing.")
                local_print_only = True
                print("Initial Sample Data (local print):", sample_data)
                break
        else:
            print("Initial Sample Data (local print):", sample_data)
    print("Successfully inserted 10 enhanced dummy sensor readings (if Firebase was available).")

    print("\nSimulating live sensor data updates. New data will be inserted every 10 seconds. Press Ctrl+C to stop.")
    while True:
        current_timestamp = datetime.now().isoformat()
        live_data = {}

        if simulation_mode:
            live_data = {
                'timestamp': current_timestamp,
                'soil_moisture': round(random.uniform(20, 80), 2),
                'temperature': round(random.uniform(20, 40), 2),
                'humidity': round(random.uniform(30, 95), 2),
                'light_intensity': random.randint(200, 900),
                'ds18b20_temperature': round(random.uniform(15, 40), 2),
                "ph": round(random.uniform(4.0, 9.0), 2),
                "N": random.randint(0, 150),
                "P": random.randint(0, 70),
                "K": random.randint(0, 250),
                "rainfall": round(random.uniform(0, 250), 2),
                "crop_stage": random.choice(CROP_STAGES),
                "growth_factor": round(random.uniform(0.1, 1.2), 2)
            }
        else:
            live_data = {
                'timestamp': current_timestamp,
                'soil_moisture': round(random.uniform(20, 60), 2),
                'temperature': round(random.uniform(20, 40), 2),
                'humidity': round(random.uniform(30, 95), 2),
                'light_intensity': random.randint(200, 900),
                'ds18b20_temperature': round(random.uniform(15, 40), 2),
                **generate_dummy_sensor_data_values()
            }

        if not local_print_only:
            try:
                firebase_db_ref.push(live_data)
            except Exception as e:
                print(f"❌ Error pushing real-time data to Firebase: {e}. Falling back to local printing.")
                local_print_only = True
                print("Live Sensor Data (local print):", live_data)
        else:
            print("Live Sensor Data (local print):", live_data)
        time.sleep(10)

# --- Helper Functions for Data Fetching and Processing ---

def get_latest_sensor_data():
    """Fetches the latest sensor data from Firebase, adding dummy values for missing fields."""
    if firebase_db_ref is None:
        print("Firebase DB reference not initialized. Cannot fetch sensor data.")
        return None

    try:
        latest_data_snapshot = firebase_db_ref.order_by_child('timestamp').limit_to_last(1).get()
        if not latest_data_snapshot:
            print("No sensor data found in Firebase. Generating dummy data for latest.")
            dummy_values = generate_dummy_sensor_data_values()
            return {
                "timestamp": datetime.now().isoformat(),
                "temperature": round(random.uniform(20, 30), 2),
                "humidity": round(random.uniform(50, 70), 2),
                "soil_moisture": round(random.uniform(40, 60), 2),
                "light_intensity": random.randint(5000, 10000),
                "ds18b20_temperature": round(random.uniform(15, 40), 2),
                **dummy_values
            }

        latest_data = list(latest_data_snapshot.values())[0]

        expected_fields = ['temperature', 'humidity', 'soil_moisture', 'light_intensity', 'ds18b20_temperature',
                            'N', 'P', 'K', 'ph', 'rainfall', 'crop_stage', 'growth_factor']
        dummy_fill_values = generate_dummy_sensor_data_values()

        for field in expected_fields:
            if field not in latest_data or latest_data[field] is None:
                latest_data[field] = dummy_fill_values.get(field)
            if isinstance(latest_data[field], (np.float64, np.int64)) and np.isnan(latest_data[field]):
                latest_data[field] = None

        if 'pH' in latest_data and 'ph' not in latest_data:
            latest_data['ph'] = latest_data['pH']
            del latest_data['pH']

        return latest_data
    except Exception as e:
        print(f"Error fetching latest sensor data from Firebase: {e}")
        dummy_values = generate_dummy_sensor_data_values()
        return {
            "timestamp": datetime.now().isoformat(),
            "temperature": round(random.uniform(20, 30), 2),
            "humidity": round(random.uniform(50, 70), 2),
            "soil_moisture": round(random.uniform(40, 60), 2),
            "light_intensity": random.randint(5000, 10000),
            "ds18b20_temperature": round(random.uniform(15, 40), 2),
            **dummy_values
        }

def get_historical_sensor_data(days=7):
    """Fetches historical sensor data from Firebase for the last 'days'."""
    if firebase_db_ref is None:
        print("Firebase DB reference not initialized. Cannot fetch historical sensor data.")
        return []
    try:
        start_time = (datetime.now() - timedelta(days=days)).isoformat()
        historical_data_snapshot = firebase_db_ref.order_by_child('timestamp').start_at(start_time).get()
        if not historical_data_snapshot:
            return []

        data_list = []
        for key, value in historical_data_snapshot.items():
            if isinstance(value, dict):
                if 'pH' in value and 'ph' not in value:
                    value['ph'] = value['pH']
                    del value['pH']

                dummy_fill_values = generate_dummy_sensor_data_values()
                for field in ['N', 'P', 'K', 'ph', 'rainfall', 'ds18b20_temperature', 'crop_stage', 'growth_factor']:
                    if field not in value or value[field] is None:
                        value[field] = dummy_fill_values.get(field)

                data_list.append(value)
            else:
                print(f"Skipping non-dict entry in Firebase historical data: {key}: {value}")

        df = pd.DataFrame(data_list)
        if df.empty:
            return []

        numeric_cols = ['N', 'P', 'K', 'ph', 'rainfall', 'temperature', 'humidity',
                        'soil_moisture', 'light_intensity', 'ds18b20_temperature', 'growth_factor']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                df[col] = np.nan
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)

        df = df.replace({np.nan: None})

        return df.to_dict(orient='records')
    except Exception as e:
        print(f"Error fetching historical sensor data from Firebase: {e}")
        return []

def fetch_camera_feed_data_backend(lang='en'):
    """Fetches the latest camera feed data (growth events) from Firebase Realtime Database and translates it."""
    if firebase_camera_ref is None:
        print("Firebase camera reference not initialized. Cannot fetch camera data.")
        return {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "stage": format_and_translate('no_camera_data', lang),
            "alert": format_and_translate('no_camera_data', lang),
            "image_url": "https://placehold.co/150x150/E0E0E0/333333?text=No+Camera+Feed"
        }

    try:
        snapshot = firebase_camera_ref.order_by_child('timestamp').limit_to_last(1).get()
        if not snapshot:
            print("No camera data found in Firebase. Returning dummy data.")
            return {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "stage": format_and_translate('no_camera_data', lang),
                "alert": format_and_translate('no_camera_data', lang),
                "image_url": "https://placehold.co/150x150/E0E0E0/333333?text=No+Camera+Feed"
            }

        latest_camera_entry = list(snapshot.values())[0]

        # Translate specific fields if they exist
        translated_stage = format_and_translate('growth_stage', lang) + ": " + translate_dynamic(latest_camera_entry.get('stage', 'N/A'), lang)
        translated_alert = format_and_translate('advisory', lang) + ": " + translate_dynamic(latest_camera_entry.get('alert', 'N/A'), lang)

        return {
            "timestamp": latest_camera_entry.get('timestamp'),
            "stage": translated_stage,
            "alert": translated_alert,
            "image_url": latest_camera_entry.get('image_url', "https://placehold.co/150x150/E0E0E0/333333?text=Camera+Feed")
        }
    except Exception as e:
        print(f"Error fetching camera feed data from Firebase: {e}")
        return {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "stage": format_and_translate('no_camera_data', lang),
            "alert": format_and_translate('no_camera_data', lang),
            "image_url": "https://placehold.co/150x150/E0E0E0/333333?text=Error+Loading+Feed"
        }

def predict_growth_backend(historical_df, selected_crop_type):
    """
    Predicts soil moisture, light intensity, and nutrient sum using the loaded AI model.
    Assumes the model was trained with specific input features and multiple outputs (time-series).
    """
    if model is None or input_scaler is None or output_scaler is None or crop_encoder is None:
        return None, None, None, "AI model, scalers, or encoder not loaded."

    LOOKBACK_WINDOW = 5

    base_sensor_features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    biological_features = ['growth_factor']

    all_tdann_input_features = base_sensor_features + biological_features

    df_for_prediction = historical_df.copy()
    dummy_values = generate_dummy_sensor_data_values()
    for col in all_tdann_input_features:
        if col not in df_for_prediction.columns or df_for_prediction[col].isnull().all():
            df_for_prediction[col] = dummy_values.get(col, 0)
        df_for_prediction[col] = pd.to_numeric(df_for_prediction[col], errors='coerce').fillna(dummy_values.get(col, 0))


    processed_data_for_prediction = df_for_prediction[all_tdann_input_features].tail(LOOKBACK_WINDOW)

    processed_data_for_prediction = processed_data_for_prediction.fillna(method='ffill').fillna(method='bfill').fillna(0)


    if len(processed_data_for_prediction) < LOOKBACK_WINDOW:
        return None, None, None, f"Not enough complete data points ({len(processed_data_for_prediction)} < {LOOKBACK_WINDOW}) after processing. Need at least {LOOKBACK_WINDOW} consecutive entries."

    try:
        crop_type_input = np.array([selected_crop_type]).reshape(-1, 1)
        encoded_crop_single = crop_encoder.transform(crop_type_input)
    except Exception as e:
        return None, None, None, f"Error encoding crop type '{selected_crop_type}': {e}. Is it in the trained labels?"

    full_input_features_sequence = []
    for i in range(LOOKBACK_WINDOW):
        numerical_features_at_timestep = processed_data_for_prediction.iloc[i][all_tdann_input_features].values
        combined_features_at_timestep = np.hstack((numerical_features_at_timestep, encoded_crop_single[0]))
        full_input_features_sequence.append(combined_features_at_timestep)

    full_input_features_sequence_np = np.array(full_input_features_sequence)

    try:
        scaled_input_sequence = input_scaler.transform(full_input_features_sequence_np)
        X_predict = scaled_input_sequence.reshape(1, LOOKBACK_WINDOW, scaled_input_sequence.shape[1])
        predicted_scaled_outputs = model.predict(X_predict, verbose=0)
        predicted_raw_outputs = output_scaler.inverse_transform(predicted_scaled_outputs)

        soil_moisture_pred = round(float(predicted_raw_outputs[0][0]), 2)
        light_intensity_pred = round(float(predicted_raw_outputs[0][1]), 2)
        nutrient_sum_pred = round(float(predicted_raw_outputs[0][2]), 2)

        return soil_moisture_pred, light_intensity_pred, nutrient_sum_pred, None
    except Exception as e:
        print(f"Error during AI prediction: {e}")
        return None, None, None, f"Error during AI prediction: {e}"

def predict_market_price_backend(latest_data, selected_crop_type):
    """
    Predicts the market price based on latest sensor data and crop type using the loaded model.
    """
    if market_price_model is None or market_crop_encoder is None or market_price_features is None:
        return None, "Market prediction model, encoder, or features not initialized."

    if not latest_data:
        return None, "No latest sensor data available for market price prediction."

    features = {}
    dummy_values = generate_dummy_sensor_data_values()
    for feature in market_price_features:
        val = latest_data.get(feature)
        if val is not None and not pd.isna(val):
            features[feature] = val
        else:
            features[feature] = dummy_values.get(feature, 0)

    input_df_numerical = pd.DataFrame([features])

    try:
        crop_type_input = np.array([selected_crop_type]).reshape(-1, 1)
        encoded_crop = market_crop_encoder.transform(crop_type_input)
        encoded_crop_df = pd.DataFrame(encoded_crop, columns=market_crop_encoder.get_feature_names_out(['crop_type']))
    except Exception as e:
        return None, f"Error encoding crop type '{selected_crop_type}' for market price: {e}"

    X_predict_market = pd.concat([input_df_numerical, encoded_crop_df], axis=1)

    expected_cols = market_price_features + market_crop_encoder.get_feature_names_out(['crop_type']).tolist()
    for col in expected_cols:
        if col not in X_predict_market.columns:
            X_predict_market[col] = 0
    X_predict_market = X_predict_market[expected_cols]

    try:
        predicted_price = market_price_model.predict(X_predict_market)[0]
        predicted_price = max(0, predicted_price)
        return round(predicted_price, 2), None
    except Exception as e:
        print(f"Error during market price prediction: {e}")
        return None, f"Error during market price prediction: {e}"

def crop_care_advice_backend(latest_data, crop_type, lang='en'):
    """Provides crop-specific care advice based on latest sensor readings."""
    if not latest_data:
        return [format_and_translate('no_data', lang)]

    tips = []
    ct = crop_type.lower()

    # NPK Advice
    npk_advice_thresholds = {
        'N': {'min': 50, 'max': 150, 'low_msg_key': 'npk_n_low', 'high_msg_key': 'npk_n_high'},
        'P': {'min': 20, 'max': 60, 'low_msg_key': 'npk_p_low', 'high_msg_key': 'npk_p_high'},
        'K': {'min': 50, 'max': 200, 'low_msg_key': 'npk_k_low', 'high_msg_key': 'npk_k_high'},
    }
    for nutrient, thresholds in npk_advice_thresholds.items():
        value = latest_data.get(nutrient)
        if value is not None and not pd.isna(value):
            if value < thresholds['min']:
                tips.append(format_and_translate('npk_low', lang, {'nutrient': nutrient, 'value': value, 'message': format_and_translate(thresholds['low_msg_key'], lang)}))
            elif value > thresholds['max']:
                tips.append(format_and_translate('npk_high', lang, {'nutrient': nutrient, 'value': value, 'message': format_and_translate(thresholds['high_msg_key'], lang)}))

    # Soil Moisture Advice
    sm = latest_data.get('soil_moisture')
    if sm is not None and not pd.isna(sm):
        if ct == 'wheat' and sm < 35: tips.append(format_and_translate('soil_moisture_low', lang, {'sm':sm, 'message':format_and_translate('wheat_sm_low', lang)}))
        elif ct == 'rice' and sm < 60: tips.append(format_and_translate('soil_moisture_low', lang, {'sm':sm, 'message':format_and_translate('rice_sm_low', lang)}))
        elif ct == 'maize' and sm < 40: tips.append(format_and_translate('soil_moisture_low', lang, {'sm':sm, 'message':format_and_translate('maize_sm_low', lang)}))
        elif ct == 'banana' and sm < 50: tips.append(format_and_translate('soil_moisture_low', lang, {'sm':sm, 'message':format_and_translate('banana_sm_low', lang)}))
        elif ct == 'mango' and sm > 60: tips.append(format_and_translate('soil_moisture_high', lang, {'sm':sm, 'message':format_and_translate('mango_sm_high', lang)}))
        elif ct == 'grapes' and sm > 50: tips.append(format_and_translate('soil_moisture_high', lang, {'sm':sm, 'message':format_and_translate('grapes_sm_high', lang)}))
        elif ct == 'cotton' and sm < 30: tips.append(format_and_translate('soil_moisture_low', lang, {'sm':sm, 'message':format_and_translate('cotton_sm_low', lang)}))
        elif (ct == 'millet' or ct == 'sorghum') and sm < 25: tips.append(format_and_translate('soil_moisture_low', lang, {'sm':sm, 'message':format_and_translate('millet_sorghum_sm_low', lang)}))
        elif ct == 'jute' and sm < 50: tips.append(format_and_translate('soil_moisture_low', lang, {'sm':sm, 'message':format_and_translate('jute_sm_low', lang)}))
        elif ct == 'pomegranate' and sm > 50: tips.append(format_and_translate('soil_moisture_high', lang, {'sm':sm, 'message':format_and_translate('pomegranate_sm_high', lang)}))
        elif (ct == 'muskmelon' or ct == 'watermelon') and sm < 30: tips.append(format_and_translate('soil_moisture_low', lang, {'sm':sm, 'message':format_and_translate('melon_sm_low', lang)}))
        elif ct == 'coconut' and sm < 50: tips.append(format_and_translate('soil_moisture_low', lang, {'sm':sm, 'message':format_and_translate('coconut_sm_low', lang)}))
        elif ct == 'mothbeans' and sm < 25: tips.append(format_and_translate('soil_moisture_low', lang, {'sm':sm, 'message':format_and_translate('mothbeans_sm_low', lang)}))
        elif ct == 'mungbean' and sm < 30: tips.append(format_and_translate('soil_moisture_low', lang, {'sm':sm, 'message':format_and_translate('mungbean_sm_low', lang)}))
        elif ct == 'blackgram' and sm < 35: tips.append(format_and_translate('soil_moisture_low', lang, {'sm':sm, 'message':format_and_translate('blackgram_sm_low', lang)}))
        elif ct == 'lentil' and sm < 25: tips.append(format_and_translate('soil_moisture_low', lang, {'sm':sm, 'message':format_and_translate('lentil_sm_low', lang)}))
        elif sm < 30: tips.append(format_and_translate('soil_moisture_low', lang, {'sm':sm, 'message':format_and_translate('general_sm_low', lang)}))
        elif sm > 70: tips.append(format_and_translate('soil_moisture_high', lang, {'sm':sm, 'message':format_and_translate('general_sm_high', lang)}))

    # Temperature Advice (combining 'temperature' and 'ds18b20_temperature')
    temp = latest_data.get('temperature')
    ds_temp = latest_data.get('ds18b20_temperature')

    effective_temp = None
    if ds_temp is not None and not pd.isna(ds_temp) and 0 <= ds_temp <= 50:
        effective_temp = ds_temp
    elif temp is not None and not pd.isna(temp):
        effective_temp = temp

    if effective_temp is not None:
        if ct == 'wheat' and effective_temp > 32: tips.append(format_and_translate('temp_high', lang, {'temp':effective_temp, 'message':format_and_translate('wheat_temp_high', lang)}))
        elif ct == 'rice' and effective_temp > 38: tips.append(format_and_translate('temp_high', lang, {'temp':effective_temp, 'message':format_and_translate('rice_temp_high', lang)}))
        elif ct == 'maize' and effective_temp < 20: tips.append(format_and_translate('temp_low', lang, {'temp':effective_temp, 'message':format_and_translate('maize_temp_low', lang)}))
        elif ct == 'banana' and effective_temp < 15: tips.append(format_and_translate('temp_low', lang, {'temp':effective_temp, 'message':format_and_translate('banana_temp_low', lang)}))
        elif ct == 'mango' and effective_temp < 20: tips.append(format_and_translate('temp_low', lang, {'temp':effective_temp, 'message':format_and_translate('mango_temp_low', lang)}))
        elif ct == 'cotton' and effective_temp < 20: tips.append(format_and_translate('temp_low', lang, {'temp':effective_temp, 'message':format_and_translate('cotton_temp_low', lang)}))
        elif (ct == 'millet' or ct == 'sorghum') and effective_temp < 20: tips.append(format_and_translate('temp_low', lang, {'temp':effective_temp, 'message':format_and_translate('millet_sorghum_temp_low', lang)}))
        elif ct == 'coffee' and effective_temp < 18: tips.append(format_and_translate('temp_low', lang, {'temp':effective_temp, 'message':format_and_translate('coffee_temp_low', lang)}))
        elif ct == 'jute' and effective_temp < 25: tips.append(format_and_translate('temp_low', lang, {'temp':effective_temp, 'message':format_and_translate('jute_temp_low', lang)}))
        elif ct == 'papaya' and effective_temp < 20: tips.append(format_and_translate('temp_low', lang, {'temp':effective_temp, 'message':format_and_translate('papaya_temp_low', lang)}))
        elif ct == 'pomegranate' and effective_temp < 20: tips.append(format_and_translate('temp_low', lang, {'temp':effective_temp, 'message':format_and_translate('pomegranate_temp_low', lang)}))
        elif (ct == 'muskmelon' or ct == 'watermelon') and effective_temp < 25: tips.append(format_and_translate('temp_low', lang, {'temp':effective_temp, 'message':format_and_translate('melon_temp_low', lang)}))
        elif ct == 'coconut' and effective_temp < 25: tips.append(format_and_translate('temp_low', lang, {'temp':effective_temp, 'message':format_and_translate('coconut_temp_low', lang)}))
        elif ct == 'mothbeans' and effective_temp < 22: tips.append(format_and_translate('temp_low', lang, {'temp':effective_temp, 'message':format_and_translate('mothbeans_temp_low', lang)}))
        elif ct == 'mungbean' and effective_temp < 20: tips.append(format_and_translate('temp_low', lang, {'temp':effective_temp, 'message':format_and_translate('mungbean_temp_low', lang)}))
        elif ct == 'blackgram' and effective_temp < 18: tips.append(format_and_translate('temp_low', lang, {'temp':effective_temp, 'message':format_and_translate('blackgram_temp_low', lang)}))
        elif ct == 'lentil' and effective_temp < 15: tips.append(format_and_translate('temp_low', lang, {'temp':effective_temp, 'message':format_and_translate('lentil_temp_low', lang)}))
        elif effective_temp < 18: tips.append(format_and_translate('temp_low', lang, {'temp':effective_temp, 'message':format_and_translate('general_temp_low', lang)}))
        elif effective_temp > 35: tips.append(format_and_translate('temp_high', lang, {'temp':effective_temp, 'message':format_and_translate('general_temp_high', lang)}))

    # Humidity Advice
    hum = latest_data.get('humidity')
    if hum is not None and not pd.isna(hum):
        if ct == 'wheat' and hum > 70: tips.append(format_and_translate('humidity_high', lang, {'hum':hum, 'message':format_and_translate('wheat_hum_high', lang)}))
        elif ct == 'rice' and hum < 60: tips.append(format_and_translate('humidity_low', lang, {'hum':hum, 'message':format_and_translate('rice_hum_low', lang)}))
        elif ct == 'banana' and hum < 60: tips.append(format_and_translate('humidity_low', lang, {'hum':hum, 'message':format_and_translate('banana_hum_low', lang)}))
        elif ct == 'grapes' and hum > 70: tips.append(format_and_translate('humidity_high', lang, {'hum':hum, 'message':format_and_translate('grapes_hum_high', lang)}))
        elif ct == 'coffee' and hum < 60: tips.append(format_and_translate('humidity_low', lang, {'hum':hum, 'message':format_and_translate('coffee_hum_low', lang)}))
        elif ct == 'orange' and hum > 70: tips.append(format_and_translate('humidity_high', lang, {'hum':hum, 'message':format_and_translate('orange_hum_high', lang)}))
        elif hum < 40: tips.append(format_and_translate('humidity_low', lang, {'hum':hum, 'message':format_and_translate('general_hum_low', lang)}))
        elif hum > 80: tips.append(format_and_translate('humidity_high', lang, {'hum':hum, 'message':format_and_translate('general_hum_high', lang)}))

    # pH Advice
    ph_val = latest_data.get('ph')
    if ph_val is not None and not pd.isna(ph_val):
        if ct == 'wheat' and ph_val < 6.0: tips.append(format_and_translate('ph_low', lang, {'ph_val':ph_val, 'message':format_and_translate('wheat_ph_low', lang)}))
        elif ct == 'rice' and (ph_val < 5.5 or ph_val > 6.5): tips.append(format_and_translate('ph_off', lang, {'ph_val':ph_val, 'message':format_and_translate('rice_ph_off', lang)}))
        elif ct == 'maize' and (ph_val < 5.8 or ph_val > 7): tips.append(format_and_translate('ph_off', lang, {'ph_val':ph_val, 'message':format_and_translate('maize_ph_off', lang)}))
        elif ct == 'papaya' and ph_val < 6: tips.append(format_and_translate('ph_low', lang, {'ph_val':ph_val, 'message':format_and_translate('papaya_ph_low', lang)}))
        elif ct == 'orange' and (ph_val < 6 or ph_val > 7.5): tips.append(format_and_translate('ph_off', lang, {'ph_val':ph_val, 'message':format_and_translate('orange_ph_off', lang)}))
        elif ph_val < 5.5: tips.append(format_and_translate('ph_low', lang, {'ph_val':ph_val, 'message':format_and_translate('general_ph_very_low', lang)}))
        elif ph_val > 7.5: tips.append(format_and_translate('ph_high', lang, {'ph_val':ph_val, 'message':format_and_translate('general_ph_very_high', lang)}))
        elif not (5.5 <= ph_val <= 7.5): tips.append(format_and_translate('ph_off', lang, {'ph_val':ph_val, 'message':format_and_translate('general_ph_off', lang)}))

    # Light Intensity Advice
    light = latest_data.get('light_intensity')
    if light is not None and not pd.isna(light):
        if ct == 'wheat' and light < 400: tips.append(format_and_translate('light_low', lang, {'light':light, 'message':format_and_translate('wheat_light_low', lang)}))
        elif ct == 'rice' and light < 500: tips.append(format_and_translate('light_low', lang, {'light':light, 'message':format_and_translate('rice_light_low', lang)}))
        elif light < 300: tips.append(format_and_translate('light_low', lang, {'light':light, 'message':format_and_translate('general_light_low', lang)}))
        elif light > 800: tips.append(format_and_translate('light_high', lang, {'light':light, 'message':format_and_translate('general_light_high', lang)}))

    # Rainfall Advice
    rain = latest_data.get('rainfall')
    if rain is not None and not pd.isna(rain):
        if rain < 50:
            tips.append(format_and_translate('rainfall_low_msg', lang, {'rain':rain, 'message':format_and_translate('rainfall_low_msg', lang)}))
        elif rain > 200:
            tips.append(format_and_translate('rainfall_high_msg', lang, {'rain':rain, 'message':format_and_translate('rainfall_high_msg', lang)}))

    return tips if tips else [format_and_translate('all_good', lang)]

def recommend_seeds_backend(soil_moisture_pred, lang='en'):
    """
    Suggests suitable crops based on predicted soil moisture.
    """
    if soil_moisture_pred is None or pd.isna(soil_moisture_pred) or not (0 <= soil_moisture_pred <= 100):
        return format_and_translate('no_specific', lang) + " (" + format_and_translate('predicted_soil_moisture_unusual', lang) + ")"
    elif soil_moisture_pred < 30:
        return f"{format_and_translate('intro', lang)} {format_and_translate('drought_resistant', lang)}{format_and_translate('outro', lang)}"
    elif soil_moisture_pred > 70:
        return f"{format_and_translate('intro', lang)} {format_and_translate('water_loving', lang)}{format_and_translate('outro', lang)}"
    else:
        return f"{format_and_translate('intro', lang)} {format_and_translate('moderate_rainfall', lang)}{format_and_translate('outro', lang)}"

def speak_tip_backend(text, lang='en'):
    """Generates speech from text using gTTS and returns audio bytes."""
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp.getvalue(), None
    except Exception as e:
        print(f"Error generating speech with gTTS for language '{lang}': {e}")
        return None, f"Error generating speech: {e}"

# --- Flask Routes ---

@app.route('/')
def index():
    return send_file('index.html')

@app.route('/api/set_mode', methods=['POST'])
def set_mode():
    """
    Endpoint to set the simulation mode.
    'simulation' mode: The backend will generate all dummy sensor data.
    'real-time' mode: The backend will generate dummy data for non-ESP32 sensors,
                      and expects real data for core sensors via /api/sensor_data.
    """
    global simulation_mode
    data = request.get_json()
    mode = data.get('mode')
    if mode == 'simulation':
        simulation_mode = True
        print("Switched to Simulation Mode (full dummy data).")
    elif mode == 'real-time':
        simulation_mode = False
        print("Switched to Real-Time Testing Mode (partial dummy, expects real).")
    else:
        return jsonify({"status": "error", "message": "Invalid mode."}), 400
    return jsonify({"status": "success", "mode": "simulation" if simulation_mode else "real-time"})

@app.route('/api/get_mode', methods=['GET'])
def get_mode():
    """Endpoint to get the current simulation mode."""
    global simulation_mode
    return jsonify({"mode": "simulation" if simulation_mode else "real-time"})


@app.route('/api/sensor_data', methods=['POST'])
def receive_sensor_data():
    """
    Receives sensor data from ESP32.
    Expected JSON: { "temperature": X, "humidity": Y, "soil_moisture": Z, "light_intensity": A, "ds18b20_temperature": B }
    pH, NPK, Rainfall, crop_stage, growth_factor are assumed to be generated as dummy if not provided.
    """
    if not firebase_db_ref:
        return jsonify({"status": "error", "message": "Firebase not initialized."}), 500

    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "Invalid JSON data."}), 400

        sensor_entry = {
            "timestamp": datetime.now().isoformat(),
            "temperature": data.get('temperature'),
            "humidity": data.get('humidity'),
            "soil_moisture": data.get('soil_moisture'),
            "light_intensity": data.get('light_intensity'),
            "ds18b20_temperature": data.get('ds18b20_temperature'),
        }

        dummy_values = generate_dummy_sensor_data_values()
        sensor_entry['ph'] = data.get('ph', dummy_values['ph'])
        sensor_entry['N'] = data.get('N', dummy_values['N'])
        sensor_entry['P'] = data.get('P', dummy_values['P'])
        sensor_entry['K'] = data.get('K', dummy_values['K'])
        sensor_entry['rainfall'] = data.get('rainfall', dummy_values['rainfall'])
        sensor_entry['crop_stage'] = data.get('crop_stage', dummy_values['crop_stage'])
        sensor_entry['growth_factor'] = data.get('growth_factor', dummy_values['growth_factor'])

        firebase_db_ref.push(sensor_entry)
        print(f"Received and stored sensor data: {sensor_entry}")

        snapshots = firebase_db_ref.order_by_child('timestamp').get()
        if snapshots and len(snapshots) > 100:
            oldest_keys = sorted(snapshots.keys(), key=lambda k: snapshots[k]['timestamp'])
            for i in range(len(oldest_keys) - 100):
                firebase_db_ref.child(oldest_keys[i]).delete()

        return jsonify({"status": "success", "message": "Sensor data received and stored."})
    except Exception as e:
        print(f"Error receiving sensor data: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/data')
def get_dashboard_data():
    """Fetches core sensor data for the dashboard."""
    lang = request.args.get('lang', 'en')

    latest_data = get_latest_sensor_data()
    historical_data_list = get_historical_sensor_data(days=7)

    camera_data = fetch_camera_feed_data_backend(lang)

    plot_data_list = []
    if historical_data_list:
        df_hist = pd.DataFrame(historical_data_list)
        df_hist['timestamp'] = pd.to_datetime(df_hist['timestamp'])
        df_hist = df_hist.sort_values(by='timestamp')

        plot_features = ['soil_moisture', 'temperature', 'humidity', 'ph', 'light_intensity', 'N', 'P', 'K', 'rainfall', 'ds18b20_temperature', 'growth_factor']
        existing_plot_features = [f for f in plot_features if f in df_hist.columns]

        if not df_hist.empty and len(existing_plot_features) > 0:
            plot_df_melted = df_hist.dropna(subset=existing_plot_features + ['timestamp']).melt(
                id_vars=['timestamp'],
                value_vars=existing_plot_features,
                var_name='Sensor Metric',
                value_name='Reading'
            )
            plot_df_melted['timestamp'] = plot_df_melted['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S')
            plot_df_melted['Reading'] = plot_df_melted['Reading'].apply(lambda x: None if pd.isna(x) else x)
            plot_data_list = plot_df_melted.to_dict(orient='records')

    raw_data_list = []
    if historical_data_list:
        df_raw = pd.DataFrame(historical_data_list).tail(10).copy()
        if 'timestamp' in df_raw.columns:
            df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
        raw_data_list = df_raw.replace({np.nan: None}).to_dict(orient='records')


    return jsonify({
        'latest_data': latest_data,
        'camera_data': camera_data,
        'plot_data': plot_data_list,
        'raw_data': raw_data_list,
        'crop_labels': all_crop_labels,
        'status': 'success' if latest_data else 'no_data'
    })


@app.route('/api/weather_data')
def get_weather_data():
    """Endpoint to fetch weather data."""
    return jsonify(generate_dummy_weather_data())


@app.route('/api/recent_events_data')
def get_recent_events_data():
    """Endpoint to fetch recent events data."""
    return jsonify(generate_dummy_recent_events())


@app.route('/api/farm_health_data')
def get_farm_health_data():
    """Endpoint to fetch farm health index data."""
    return jsonify(generate_dummy_farm_health_data())


@app.route('/api/device_connectivity_data')
def get_device_connectivity_data():
    """Endpoint to fetch device connectivity data."""
    return jsonify(generate_dummy_device_connectivity())


@app.route('/api/resource_consumption_data')
def get_resource_consumption_data():
    """Endpoint to fetch resource consumption data."""
    return jsonify(generate_dummy_resource_consumption())


@app.route('/api/usage_history')
def get_usage_history_data():
    """Endpoint to fetch historical resource consumption data."""
    return jsonify(generate_dummy_usage_history_data())


@app.route('/api/pest_scan_trigger', methods=['POST'])
def api_pest_scan_trigger():
    """Endpoint to simulate triggering a pest scan and return results."""
    time.sleep(2)
    results = generate_dummy_pest_scan_results()
    return jsonify(results)


@app.route('/api/action', methods=['POST'])
def api_quick_action():
    """Endpoint for quick actions like irrigation, nutrient application, alerts."""
    data = request.get_json()
    action_type = data.get('action_type')

    if action_type == 'irrigation':
        print(f"Action: Initiating Irrigation at {datetime.now()}")
        message = "Irrigation initiated successfully."
    elif action_type == 'nutrients':
        print(f"Action: Applying Nutrients at {datetime.now()}")
        message = "Nutrients applied successfully."
    elif action_type == 'alert':
        alert_message = data.get('message', 'General alert from dashboard.')
        print(f"Action: Sending Alert: '{alert_message}' at {datetime.now()}")
        message = f"Alert '{alert_message}' sent successfully."
    else:
        return jsonify({"status": "error", "message": "Invalid action type."}), 400

    return jsonify({"status": "success", "message": message})


@app.route('/api/predict_growth', methods=['POST'])
def api_predict_growth():
    data = request.get_json()
    selected_crop_type = data.get('selected_crop_type')

    historical_data_dicts = get_historical_sensor_data(days=7)
    if not historical_data_dicts:
        return jsonify({'error': 'No sensor data available for prediction. Please send data to /api/sensor_data.'}), 400

    df_historical = pd.DataFrame(historical_data_dicts)
    if 'pH' in df_historical.columns and 'ph' not in df_historical.columns:
        df_historical['ph'] = df_historical['pH']
        df_historical = df_historical.drop(columns=['pH'])

    soil_moisture_pred, light_intensity_pred, nutrient_sum_pred, error_msg = predict_growth_backend(df_historical, selected_crop_type)

    if error_msg:
        return jsonify({'error': error_msg}), 500

    return jsonify({
        'soil_moisture_pred': soil_moisture_pred,
        'light_intensity_pred': light_intensity_pred,
        'nutrient_sum_pred': nutrient_sum_pred
    })


@app.route('/api/market_price', methods=['POST'])
def api_market_price():
    data = request.get_json()
    selected_crop_type = data.get('selected_crop_type')

    latest_sensor_data = get_latest_sensor_data()
    if not latest_sensor_data:
        return jsonify({'error': 'No sensor data available for market price prediction.'}), 400

    predicted_price, error_msg = predict_market_price_backend(latest_sensor_data, selected_crop_type)

    if error_msg:
        return jsonify({'error': error_msg}), 500

    return jsonify({'predicted_price': predicted_price})


@app.route('/api/care_advice', methods=['POST'])
def api_care_advice():
    data = request.get_json()
    selected_crop_type = data.get('selected_crop_type')
    lang = data.get('lang', 'en')

    latest_data = get_latest_sensor_data()
    if not latest_data:
        return jsonify({'advice': [format_and_translate('no_data', lang)]})

    care_tips = crop_care_advice_backend(latest_data, selected_crop_type, lang)
    return jsonify({'advice': care_tips})


@app.route('/api/seed_recommendations', methods=['POST'])
def api_seed_recommendations():
    data = request.get_json()
    soil_moisture_pred = data.get('soil_moisture_pred')
    lang = data.get('lang', 'en')

    seed_recommendation = recommend_seeds_backend(soil_moisture_pred, lang)
    return jsonify({'recommendation': seed_recommendation})


@app.route('/api/voice_alert', methods=['POST'])
def api_voice_alert():
    data = request.get_json()
    text = data.get('text')
    lang = data.get('lang', 'en')

    if not text:
        return jsonify({'error': 'No text provided for speech generation.'}), 400

    audio_bytes, error_msg = speak_tip_backend(text, lang)
    if error_msg:
        return jsonify({'error': error_msg}), 500

    return send_file(
        io.BytesIO(audio_bytes),
        mimetype='audio/mpeg',
        as_attachment=False,
        download_name='alert.mp3'
    )


@app.route('/api/crop_labels')
def get_crop_labels():
    return jsonify({'crop_labels': all_crop_labels})


@app.route('/languages', methods=['GET'])
def get_libretranslate_languages():
    """Fetches supported languages directly from LibreTranslate."""
    try:
        response = requests.get(f"{LIBRE_TRANSLATE_BASE_URL}/languages")
        response.raise_for_status()
        return jsonify(response.json())
    except requests.exceptions.RequestException as e:
        print(f"Error fetching languages from LibreTranslate: {e}")
        return jsonify({"error": "Failed to fetch languages from translation service.", "details": str(e)}), 500


@app.route('/api/translate_static_text', methods=['POST'])
def translate_static_text_api():
    """
    Translates a list of static UI text keys using the format_and_translate helper.
    """
    data = request.get_json()
    keys = data.get('keys', [])
    target_lang = data.get('target_lang', 'en')
    source_lang = data.get('source_lang', 'en')

    translated_texts = {}
    for key in keys:
        # Use format_and_translate with an empty dict for variables, as these are static keys
        translated_texts[key] = format_and_translate(key, target_lang, {}, source_lang)

    return jsonify(translated_texts)


# --- Data Export Endpoints ---

@app.route('/api/export_csv', methods=['GET'])
def export_csv():
    """Exports historical sensor data as a CSV file."""
    historical_data = get_historical_sensor_data(days=365)
    if not historical_data:
        return jsonify({"error": "No data available to export."}), 404

    df = pd.DataFrame(historical_data)
    df = df.astype(str).replace('None', '')

    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    return Response(
        csv_buffer.getvalue(),
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=sensor_data.csv"}
    )


@app.route('/api/export_excel', methods=['GET'])
def export_excel():
    """Exports historical sensor data as an Excel file."""
    historical_data = get_historical_sensor_data(days=365)
    if not historical_data:
        return jsonify({"error": "No data available to export."}), 404

    df = pd.DataFrame(historical_data)
    df = df.astype(str).replace('None', '')

    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sensor Data')
    excel_buffer.seek(0)

    return Response(
        excel_buffer.getvalue(),
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-disposition": "attachment; filename=sensor_data.xlsx"}
    )


@app.before_request
def before_first_request():
    if not firebase_app:
        initialize_firebase()
    if model is None or input_scaler is None or output_scaler is None or crop_encoder is None or market_price_model is None:
        load_models_and_scalers()


if __name__ == '__main__':
    with app.app_context():
        initialize_firebase()
        load_models_and_scalers()

    camera_thread = threading.Thread(target=run_camera_simulator_thread)
    camera_thread.daemon = True
    camera_thread.start()

    sensor_inserter_thread = threading.Thread(target=run_sensor_data_inserter_thread)
    sensor_inserter_thread.daemon = True
    sensor_inserter_thread.start()

    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
