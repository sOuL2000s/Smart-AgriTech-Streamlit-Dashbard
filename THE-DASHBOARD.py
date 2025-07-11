import random
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
import firebase_admin
from firebase_admin import credentials, db
import datetime
import plotly.express as px
import plotly.graph_objects as go # Import for gauge charts
import base64
import tempfile
import os
import json
import joblib # For saving/loading scalers
import time
from streamlit_autorefresh import st_autorefresh

# For Voice Alerts
from gtts import gTTS

# --- Firebase Secure Setup (Render-Compatible) ---
firebase_key_b64 = os.getenv("FIREBASE_KEY_B64")
firebase_cred_path = None # Initialize to None

firebase_init_status = []

try:
    if firebase_key_b64:
        decoded_json = base64.b64decode(firebase_key_b64).decode('utf-8')
        with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.json') as f:
            f.write(decoded_json)
        firebase_cred_path = f.name
        cred = credentials.Certificate(firebase_cred_path)
    else:
        # Fallback for local development if environment variable is not set
        # Ensure 'agriastrax-website-firebase-adminsdk-fbsvc-36cdff39c2.json' is in your project root
        cred = credentials.Certificate("agriastrax-website-firebase-adminsdk-fbsvc-36cdff39c2.json")

    # Prevent double initialization
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://agriastrax-website-default-rtdb.firebaseio.com/'
        })
    firebase_init_status.append("✅ Firebase initialized successfully.")
except Exception as e:
    firebase_init_status.append(f"❌ Firebase initialization failed: {e}")
    st.error(f"❌ Firebase initialization failed: {e}", icon="❌")
    st.stop() # Stop the app if Firebase fails to initialize
finally:
    # Clean up the temporary file if created from environment variable
    if firebase_key_b64 and firebase_cred_path and os.path.exists(firebase_cred_path):
        os.remove(firebase_cred_path)

# --- Load Real Crop Labels from CSV ---
all_crop_labels = []
crop_encoder = None
try:
    crop_df_for_labels = pd.read_csv("cleaned_sensor_data.csv")
    all_crop_labels = sorted(crop_df_for_labels['label'].unique().tolist())
    
    # Initialize OneHotEncoder for crop type for consistent encoding
    crop_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    crop_encoder.fit(np.array(all_crop_labels).reshape(-1, 1)) # Fit with all known crop labels
    firebase_init_status.append(f"✅ Crop labels loaded: {len(all_crop_labels)} unique crops found.")
except FileNotFoundError:
    firebase_init_status.append("❌ 'cleaned_sensor_data.csv' not found. Please ensure it's in the same directory.")
    st.error("❌ 'cleaned_sensor_data.csv' not found. Please ensure it's in the same directory.", icon="❌")
    all_crop_labels = [] # Initialize as empty to prevent errors later
    # Fallback encoder, might not be fully representative without actual data
    crop_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False) 
except Exception as e:
    firebase_init_status.append(f"❌ Error loading 'cleaned_sensor_data.csv': {e}")
    st.error(f"❌ Error loading 'cleaned_sensor_data.csv': {e}", icon="❌")
    all_crop_labels = []
    # Fallback encoder
    crop_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False) 

# --- Load AI Model ---
model = None
try:
    model = tf.keras.models.load_model("tdann_pnsm_model.keras")
    firebase_init_status.append("✅ AI model (tdann_pnsm_model.keras) loaded successfully.")
except Exception as e:
    firebase_init_status.append(f"❌ Error loading AI model (tdann_pnsm_model.keras): {e}")
    st.error(f"❌ Error loading AI model (tdann_pnsm_model.keras): {e}", icon="❌")
    st.stop() # Stop the app if the model cannot be loaded

# --- Load Scalers ---
# IMPORTANT: These scalers MUST be the ones fitted during the model training phase.
input_scaler = None
output_scaler = None
try:
    input_scaler = joblib.load('tdann_input_scaler.joblib')
    output_scaler = joblib.load('tdann_output_scaler.joblib')
    firebase_init_status.append("✅ Input and Output scalers loaded successfully.")
except FileNotFoundError:
    firebase_init_status.append("❌ Scaler files (tdann_input_scaler.joblib, tdann_output_scaler.joblib) not found. "
             "The model predictions might be inaccurate without the correct scalers. "
             "Please ensure they are saved during model training and placed in the same directory.")
    st.error("❌ Scaler files (tdann_input_scaler.joblib, tdann_output_scaler.joblib) not found. "
             "The model predictions might be inaccurate without the correct scalers. "
             "Please ensure they are saved during model training and placed in the same directory.", icon="❌")
    # In a real production environment, you might want to stop the app here or handle robustly.
    input_scaler = MinMaxScaler() # Fallback: Initialize new scalers, but warn the user.
    output_scaler = MinMaxScaler() # Fallback: Initialize new scalers, but warn the user.
    st.warning("⚠️ Proceeding with newly initialized scalers. Predictions may be inaccurate.", icon="⚠️")
except Exception as e:
    firebase_init_status.append(f"❌ Error loading scalers: {e}")
    st.error(f"❌ Error loading scalers: {e}", icon="❌")
    input_scaler = MinMaxScaler() # Fallback
    output_scaler = MinMaxScaler() # Fallback
    st.warning("⚠️ Proceeding with newly initialized scalers. Predictions may be inaccurate.", icon="⚠️")


# --- Market Price Predictor Setup ---
# Simulate Market Price Data
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
        
        base_price = 100
        
        if crop_type == 'wheat':
            price = base_price * 1.2
        elif crop_type == 'rice':
            price = base_price * 1.5
        elif crop_type == 'maize':
            price = base_price * 1.1
        else: 
            price = base_price * 1.0
            
        price += (N / 10) + (P / 5) + (K / 10)
        price += (temperature - 25) * 2
        price += (humidity - 50) * 1.5
        
        price += random.uniform(-10, 10)
        price = max(50, price)
        
        data.append([N, P, K, temperature, humidity, crop_type, price])
        
    df_prices = pd.DataFrame(data, columns=['N', 'P', 'K', 'temperature', 'humidity', 'crop_type', 'price'])
    return df_prices

@st.cache_data 
def train_market_price_model():
    if crop_encoder is None:
        st.error("Cannot train market price model: Crop encoder not initialized.", icon="❌")
        return None, None, None

    df_prices = generate_market_price_data(num_samples=2000)
    
    market_price_features = ['N', 'P', 'K', 'temperature', 'humidity']
    
    # Ensure crop_encoder is fitted with 'crop_type' if it's used here
    # The `all_crop_labels` used to fit the main crop_encoder should be sufficient.
    X_categorical = crop_encoder.transform(df_prices[['crop_type']])
    X_categorical_df = pd.DataFrame(X_categorical, columns=crop_encoder.get_feature_names_out(['crop_type']))
    
    X_numerical = df_prices[market_price_features]
    
    X_train_market = pd.concat([X_numerical, X_categorical_df], axis=1)
    y_train_market = df_prices['price']
    
    market_model = LinearRegression()
    market_model.fit(X_train_market, y_train_market)
    
    return market_model, crop_encoder, market_price_features

market_price_model, market_crop_encoder, market_price_features = train_market_price_model()
if market_price_model:
    firebase_init_status.append("✅ Market price prediction model trained (simulated data).")
else:
    firebase_init_status.append("❌ Market price prediction model could not be trained.")
    st.error("❌ Market price prediction model could not be trained.", icon="❌")


# --- Fetch Live Sensor Data ---
@st.cache_data(ttl=10) # Cache data for 10 seconds to reduce Firebase reads
def fetch_sensor_data():
    """Fetches sensor data from Firebase Realtime Database."""
    ref = db.reference('sensors/farm1')
    snapshot = ref.get()
    if not snapshot:
        return pd.DataFrame()
    
    # Firebase data often comes as a dict of dicts, where keys are timestamps
    # Convert to a list of dicts, then DataFrame
    if isinstance(snapshot, dict):
        data_list = []
        for key, value in snapshot.items():
            if isinstance(value, dict):
                data_list.append(value)
            else:
                st.warning(f"Skipping non-dict entry in Firebase: {key}: {value}", icon="⚠️")
        df = pd.DataFrame(data_list)
    else: # If snapshot is already a list of dicts or single dict
        df = pd.DataFrame(snapshot)

    if df.empty:
        return pd.DataFrame()

    # Convert relevant columns to numeric, coercing errors
    # Note: 'pH' is used here as per Firebase, 'ph' was used in training script
    # Ensure consistency or handle both if they might appear.
    numeric_cols = ['N', 'P', 'K', 'pH', 'rainfall', 'temperature', 'humidity', 
                    'soil_moisture', 'light_intensity', 'growth_factor', 'growth_trigger']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            df[col] = np.nan # Ensure missing columns are added as NaN to align with expected features
    
    # Handle 'pH' column name consistency: Firebase might send 'pH', trainer expects 'ph'
    # Do this early to ensure subsequent processing uses 'ph'
    if 'pH' in df.columns and 'ph' not in df.columns:
        df['ph'] = df['pH']
    if 'pH' in df.columns: # Optionally drop the 'pH' column if 'ph' is preferred
        df = df.drop(columns=['pH'])

    # --- NEW: Impute specific problematic columns if NaN with a sensible default ---
    # Apply this AFTER initial numeric conversion and pH/ph mapping
    if 'ph' in df.columns and df['ph'].isnull().any():
        # Use a common neutral pH (6.5) as default if NaN
        df['ph'] = df['ph'].fillna(6.5) 
        # Removed st.warning here to make it less intrusive
    # --- END NEW ---

    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    df = df.sort_values('timestamp')
    return df.reset_index(drop=True)

# --- Fetch Live Camera Feed Data ---
@st.cache_data(ttl=10) # Cache data for 10 seconds
def fetch_camera_feed_data():
    """Fetches camera feed data (growth events) from Firebase Realtime Database."""
    ref = db.reference('camera_feed/farm1')
    snapshot = ref.get()
    if not snapshot:
        return None
    
    # Firebase data often comes as a dict of dicts, where keys are Firebase push IDs
    # Get the last entry by sorting by key (which are often timestamp-like or sequential)
    if isinstance(snapshot, dict):
        # Get the last key (latest entry)
        last_key = sorted(snapshot.keys())[-1]
        return snapshot[last_key]
    else:
        # This case should ideally not happen for push data, but handle defensively
        if isinstance(snapshot, list) and snapshot:
            return snapshot[-1] # Get the last item if it's a list
        return None


# --- Predict Growth (Multi-Output TDANN) ---
def predict_growth(df_latest_data, selected_crop_type):
    """
    Predicts soil moisture, light intensity, and nutrient sum using the loaded AI model.
    Assumes the model was trained with specific input features and multiple outputs.
    """
    if model is None or input_scaler is None or output_scaler is None or crop_encoder is None:
        st.error("AI model or scalers or encoder not loaded. Cannot predict growth.", icon="❌")
        return None, None, None
    
    LOOKBACK_WINDOW = 5
    
    # Define input features as per trainer script
    base_sensor_features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    biological_features = ['growth_factor'] # 'growth_trigger' was removed from trainer
    
    # Determine which biological features are actually present in the data
    # Filter out features that are not in the dataframe's columns
    present_biological_features = [f for f in biological_features if f in df_latest_data.columns]

    final_tdann_input_features = base_sensor_features + present_biological_features

    # Ensure all columns in final_tdann_input_features are in df_latest_data before subsetting
    available_tdann_features = [f for f in final_tdann_input_features if f in df_latest_data.columns]
    
    if len(available_tdann_features) != len(final_tdann_input_features):
        missing = set(final_tdann_input_features) - set(available_tdann_features)
        st.error(f"Missing expected TDANN input features in sensor data: {missing}. Cannot predict growth.", icon="❌")
        st.info("Ensure Firebase is sending all required sensor data: N, P, K, temperature, humidity, pH/ph, rainfall, growth_factor.", icon="ℹ️")
        return None, None, None

    # Try to get enough data for the lookback window.
    # Use fillna() with ffill() to propagate last valid observation forward for missing values.
    # Use bfill() to fill leading NaNs if ffill has nothing to propagate.
    # IMPORTANT: df_latest_data is already pre-processed by fetch_sensor_data including pH imputation.
    processed_data_for_prediction = df_latest_data[available_tdann_features].tail(LOOKBACK_WINDOW)
    
    # A final fillna(0) as a safeguard, for any remaining NaNs, or if a whole row is NaN initially.
    # It's better to ensure data quality at the source or during fetch_sensor_data.
    processed_data_for_prediction = processed_data_for_prediction.fillna(method='ffill').fillna(method='bfill').fillna(0)


    if len(processed_data_for_prediction) < LOOKBACK_WINDOW:
        st.info(f"Not enough complete data points ({len(processed_data_for_prediction)} < {LOOKBACK_WINDOW}) even after filling NaNs. Need at least {LOOKBACK_WINDOW} consecutive entries with non-NaNs initially.", icon="ℹ️")
        st.info("Please ensure enough historical sensor data is available in Firebase for the lookback window.", icon="ℹ️")
        return None, None, None

    # Get encoded crop features column names (from the model's crop_encoder)
    encoded_crop_feature_names = crop_encoder.get_feature_names_out(['label']) # Changed from 'crop_type' to 'label' to match trainer
    
    # Combine all input features that the TDANN model expects
    # This must match the order and number of features during model training
    expected_full_input_features_order = final_tdann_input_features + encoded_crop_feature_names.tolist()

    full_input_features_sequence = []
    
    # One-hot encode the selected crop type once
    crop_type_input = np.array([selected_crop_type]).reshape(-1, 1)
    encoded_crop_single = crop_encoder.transform(crop_type_input)

    for i in range(LOOKBACK_WINDOW):
        # Get numerical features for the current timestep in the window
        numerical_features_at_timestep = processed_data_for_prediction.iloc[i][available_tdann_features].values
        
        # Combine numerical features with the *static* one-hot encoded crop type
        # Assuming crop type is static for the entire prediction window
        combined_features_at_timestep = np.hstack((numerical_features_at_timestep, encoded_crop_single[0]))
        full_input_features_sequence.append(combined_features_at_timestep)

    full_input_features_sequence_np = np.array(full_input_features_sequence)

    # --- Debugging: Print shapes and feature lists ---
    # st.write(f"DEBUG: Scaler expects {input_scaler.n_features_in_} features.")
    # st.write(f"DEBUG: Current input sequence shape: {full_input_features_sequence_np.shape}")
    # st.write(f"DEBUG: Expected full input features order: {expected_full_input_features_order}")

    # Scale the combined input features
    scaled_input_sequence = input_scaler.transform(full_input_features_sequence_np)
    
    # Reshape for LSTM input: (1, lookback, num_features)
    X_predict = scaled_input_sequence.reshape(1, LOOKBACK_WINDOW, scaled_input_sequence.shape[1])
    
    try:
        predicted_scaled_outputs = model.predict(X_predict)
        
        # Inverse transform the predictions to get original scale
        predicted_raw_outputs = output_scaler.inverse_transform(predicted_scaled_outputs)
        
        soil_moisture_pred = round(float(predicted_raw_outputs[0][0]), 2)
        light_intensity_pred = round(float(predicted_raw_outputs[0][1]), 2)
        nutrient_sum_pred = round(float(predicted_raw_outputs[0][2]), 2) # Assuming NPK sum is the third output
        
        return soil_moisture_pred, light_intensity_pred, nutrient_sum_pred
    except Exception as e:
        st.error(f"Error during AI prediction: {e}", icon="❌")
        st.exception(e) # Display full traceback for debugging
        return None, None, None

# --- Predict Market Price ---
def predict_market_price(latest_data, selected_crop_type, market_model, market_crop_encoder, market_price_features):
    """
    Predicts the market price based on latest sensor data and crop type.
    """
    if market_model is None or market_crop_encoder is None:
        st.error("Market prediction model or encoder not initialized.", icon="❌")
        return None

    if not latest_data:
        return None

    # Prepare input for the market price model
    input_values = {}
    for feature in market_price_features:
        # Use .get() and check for pd.isna as Firebase data might not have all keys or they might be NaN
        val = latest_data.get(feature)
        if val is not None and not pd.isna(val):
            input_values[feature] = val
        else:
            st.warning(f"Missing or NaN feature '{feature}' for market price prediction. Imputing with 0.", icon="⚠️")
            input_values[feature] = 0 # Impute with 0 for market price model if missing
    
    input_df_numerical = pd.DataFrame([input_values])
    
    # One-hot encode crop type for prediction
    crop_type_input = np.array([selected_crop_type]).reshape(-1, 1)
    encoded_crop = market_crop_encoder.transform(crop_type_input)
    # Ensure column names match what the market model was trained on
    encoded_crop_df = pd.DataFrame(encoded_crop, columns=market_crop_encoder.get_feature_names_out(['crop_type']))
    
    # Combine all features for prediction
    X_predict_market = pd.concat([input_df_numerical, encoded_crop_df], axis=1)
    
    try:
        predicted_price = market_model.predict(X_predict_market)[0]
        return round(predicted_price, 2)
    except Exception as e:
        st.error(f"Error during market price prediction: {e}", icon="❌")
        st.exception(e) # Display full traceback for debugging
        return None


# --- Voice Alert Function (Updated for Streamlit Cloud + Local) ---
def speak_tip(tip_text, lang='en'):
    file_path = None # Initialize file_path
    try:
        with st.spinner(f"Generating voice alert in {lang.upper()}..."):
            tts = gTTS(text=tip_text, lang=lang)
            # Use NamedTemporaryFile to get a unique file path
            # delete=False means we'll manually delete it later
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                file_path = f.name
                tts.save(file_path)
            
            # Read the audio file bytes and then close it immediately
            with open(file_path, "rb") as audio_file:
                audio_bytes = audio_file.read()
            
            st.audio(audio_bytes, format="audio/mp3", autoplay=True)
            
            # Add a small delay to give Streamlit/browser time to process the audio
            time.sleep(0.5) 

    except Exception as e:
        st.error(f"Error generating or playing voice alert: {e}", icon="❌")
        st.info("This might be due to temporary file access issues or an unsupported audio backend.", icon="ℹ️")
    finally:
        # Ensure the file is deleted only if it exists and after a short delay
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path) # Clean up the temporary file
            except Exception as e_del:
                st.warning(f"Could not delete temporary audio file {file_path}: {e_del}", icon="⚠️")


# --- Crop Care Advice Function ---
# Mapping for advice messages to support multiple languages
ADVICE_MESSAGES = {
    'en': {
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
        'general_light_high': "General advice: Excessive light can cause scorching. Consider shading during peak hours."
    },
    'hi': {
        'no_data': "सलाह देने के लिए कोई सेंसर डेटा उपलब्ध नहीं है।",
        'npk_low': "🌱 **{nutrient} कम है ({value:.1f})**: {message}",
        'npk_high': "🌱 **{nutrient} अधिक है ({value:.1f})**: {message}",
        'soil_moisture_low': "💧 **मिट्टी की नमी कम है ({sm:.1f}%)**: {message}",
        'soil_moisture_high': "💧 **मिट्टी की नमी अधिक है ({sm:.1f}%)**: {message}",
        'temp_low': "🌡️ **तापमान कम है ({temp:.1f}°C)**: {message}",
        'temp_high': "🌡️ **तापमान अधिक है ({temp:.1f}°C)**: {message}",
        'humidity_low': "💨 **आर्द्रता कम है ({hum:.1f}%)**: {message}",
        'humidity_high': "💨 **आर्द्रता अधिक है ({hum:.1f}%)**: {message}",
        'ph_low': "🧪 **पीएच कम है ({ph_val:.1f})**: {message}",
        'ph_high': "🧪 **पीएच अधिक है ({ph_val:.1f})**: {message}",
        'ph_off': "🧪 **पीएच सही नहीं है ({ph_val:.1f})**: {message}",
        'light_low': "☀️ **प्रकाश की तीव्रता कम है ({light:.1f} लक्स)**: {message}",
        'light_high': "☀️ **प्रकाश की तीव्रता अधिक है ({light:.1f} लक्स)**: {message}",
        'rainfall_low_msg': "🌧️ **वर्षा कम है ({rain:.1f} मिमी)**: {message}",
        'rainfall_high_msg': "🌧️ **वर्षा अधिक है ({rain:.1f} मिमी)**: {message}",
        'all_good': "✅ सभी मुख्य पैरामीटर ठीक दिख रहे हैं! इष्टतम विकास के लिए नियमित रूप से निगरानी करते रहें।",
        'npk_n_low': "नाइट्रोजन युक्त उर्वरक डालने पर विचार करें।",
        'npk_n_high': "अतिरिक्त नाइट्रोजन फल/फूल के विकास के बजाय पत्तों के विकास को बढ़ावा दे सकता है।",
        'npk_p_low': "जड़ के विकास के लिए फास्फोरस उर्वरक डालने पर विचार करें।",
        'npk_p_high': "उच्च फास्फोरस अन्य पोषक तत्वों को अवरुद्ध कर सकता है।",
        'npk_k_low': "पौधे के समग्र स्वास्थ्य और फल की गुणवत्ता के लिए पोटेशियम उर्वरक डालने पर विचार करें।",
        'npk_k_high': "अतिरिक्त पोटेशियम कैल्शियम और मैग्नीशियम के अवशोषण में हस्तक्षेप कर सकता है।",
        'wheat_sm_low': "हल्की सिंचाई करें – गेहूं को 35-50% मिट्टी की नमी की आवश्यकता होती है।",
        'rice_sm_low': "चावल को अधिक नमी की आवश्यकता होती है। उचित सिंचाई सुनिश्चित करें।",
        'maize_sm_low': "मक्का को मध्यम मिट्टी की नमी के स्तर की आवश्यकता होती है।",
        'banana_sm_low': "केले के लिए मिट्टी को लगातार नम रखें।",
        'mango_sm_high': "जलभराव से बचें। आम को अच्छी जल निकासी वाली मिट्टी की आवश्यकता होती है।",
        'grapes_sm_high': "अंगूर सूखी मिट्टी पसंद करते हैं – अधिक पानी देने से बचें।",
        'cotton_sm_low': "कपास को फूल आने के दौरान मध्यम नमी की आवश्यकता होती है।",
        'millet_sorghum_sm_low': "ये सूखे प्रतिरोधी फसलें हैं लेकिन फिर भी न्यूनतम नमी की आवश्यकता होती है।",
        'jute_sm_low': "जूट को विकास के दौरान पर्याप्त नमी की आवश्यकता होती है।",
        'pomegranate_sm_high': "अनार को अधिक पानी देने से बचें।",
        'melon_sm_low': "तरबूज को लगातार पानी की आवश्यकता होती है, खासकर फल लगने के दौरान।",
        'coconut_sm_low': "नारियल के पेड़ों को उच्च नमी के स्तर की आवश्यकता होती है।",
        'mothbeans_sm_low': "मोठबीन सूखे को सहन करने वाली फसलें हैं लेकिन फूल आने के दौरान न्यूनतम सिंचाई की आवश्यकता होती है।",
        'mungbean_sm_low': "फूल आने और फली बनने के दौरान नियमित सिंचाई सुनिश्चित करें।",
        'blackgram_sm_low': "विशेष रूप से फूल आने के दौरान मध्यम नमी बनाए रखें।",
        'lentil_sm_low': "मसूर को कम से मध्यम नमी की आवश्यकता होती है।",
        'general_sm_low': "सामान्य सलाह: सूखे के तनाव को रोकने के लिए सिंचाई पर विचार करें।",
        'general_sm_high': "सामान्य सलाह: जलभराव को रोकने के लिए अच्छी जल निकासी सुनिश्चित करें।",
        'wheat_temp_high': "शाम को छाया प्रदान करें या सिंचाई करें – गेहूं के लिए तापमान बहुत अधिक है।",
        'rice_temp_high': "चावल के लिए बहुत गर्म है। शाम को सिंचाई या छाया पर विचार करें।",
        'maize_temp_low': "मक्का गर्म मौसम (20-30°C) पसंद करता है।",
        'banana_temp_low': "केला ठंड के प्रति संवेदनशील है – गर्म स्थिति सुनिश्चित करें।",
        'mango_temp_low': "आम को गर्म तापमान (>20°C) की आवश्यकता होती है।",
        'cotton_temp_low': "कपास गर्म तापमान में पनपती है।",
        'millet_sorghum_temp_low': "बाजरा/ज्वार के लिए गर्म जलवायु आदर्श है।",
        'coffee_temp_low': "कॉफी 18-24°C रेंज में पनपती है।",
        'jute_temp_low': "जूट 25-30°C में अच्छी तरह उगता है।",
        'papaya_temp_low': "पपीता 21-33°C रेंज पसंद करता है।",
        'pomegranate_temp_low': "आदर्श तापमान 20°C से ऊपर है।",
        'melon_temp_low': "सुनिश्चित करें कि तापमान गर्म (>25°C) हो।",
        'coconut_temp_low': "नारियल के लिए आदर्श तापमान 25°C से ऊपर है।",
        'mothbeans_temp_low': "तापमान 22°C से ऊपर होना चाहिए।",
        'mungbean_temp_low': "मूंग को इष्टतम विकास के लिए गर्म परिस्थितियों की आवश्यकता होती है।",
        'blackgram_temp_low': "आदर्श तापमान सीमा 25-35°C है।",
        'lentil_temp_low': "मसूर 18-30°C में अच्छी तरह उगती है।",
        'general_temp_low': "सामान्य सलाह: ठंडा तापमान विकास को रोक सकता है। सुरक्षात्मक उपायों पर विचार करें।",
        'general_temp_high': "सामान्य सलाह: उच्च तापमान से गर्मी का तनाव हो सकता है। पर्याप्त पानी और छाया सुनिश्चित करें।",
        'wheat_hum_high': "कवक संक्रमण से सावधान रहें – वायु प्रवाह सुनिश्चित करें।",
        'rice_hum_low': "आसपास की आर्द्रता बढ़ाएँ या पलवार का उपयोग करें।",
        'banana_hum_low': "केले को उच्च आर्द्रता की आवश्यकता होती है। धुंध या पलवार पर विचार करें।",
        'grapes_hum_high': "उच्च आर्द्रता से कवक संक्रमण हो सकता है।",
        'coffee_hum_low': "कॉफी उच्च आर्द्रता पसंद करती है।",
        'orange_hum_high': "वायु प्रवाह में सुधार और कवक संबंधी समस्याओं को रोकने के लिए पेड़ों की छंटाई करें।",
        'general_hum_low': "सामान्य सलाह: कम आर्द्रता से मुरझाना हो सकता है। धुंध या मिट्टी की नमी बढ़ाने पर विचार करें।",
        'general_hum_high': "सामान्य सलाह: उच्च आर्द्रता से कवक रोगों का खतरा बढ़ जाता है। अच्छा वेंटिलेशन सुनिश्चित करें।",
        'wheat_ph_low': "थोड़ा अम्लीय – पीएच बढ़ाने के लिए चूना डालने पर विचार करें।",
        'rice_ph_off': "चावल के लिए थोड़ी अम्लीय मिट्टी बनाए रखें (पीएच 5.5-6.5)।",
        'maize_ph_off': "मिट्टी का पीएच 5.8-7.0 के बीच बनाए रखें।",
        'papaya_ph_low': "पपीते के लिए थोड़ी अम्लीय से तटस्थ मिट्टी सबसे अच्छी होती है।",
        'orange_ph_off': "संतरे के लिए आदर्श मिट्टी का पीएच 6.0-7.5 है।",
        'general_ph_very_low': "सामान्य सलाह: मिट्टी बहुत अम्लीय है। पीएच बढ़ाने और पोषक तत्वों की उपलब्धता में सुधार के लिए चूना डालें।",
        'general_ph_very_high': "सामान्य सलाह: मिट्टी बहुत क्षारीय है। पीएच कम करने के लिए सल्फर या जैविक पदार्थ डालें।",
        'general_ph_off': "सामान्य सलाह: अधिकांश फसलों के लिए इष्टतम पीएच रेंज 5.5-7.5 है। आवश्यकतानुसार समायोजित करें।",
        'wheat_light_low': "सुनिश्चित करें कि फसल को पर्याप्त धूप मिले।",
        'rice_light_low': "सुनिश्चित करें कि चावल को पूरी धूप मिले।",
        'general_light_low': "सामान्य सलाह: अपर्याप्त प्रकाश प्रकाश संश्लेषण में बाधा डाल सकता है। पूरक प्रकाश या छंटाई पर विचार करें।",
        'general_light_high': "सामान्य सलाह: अत्यधिक प्रकाश से झुलसना हो सकता है। चरम घंटों के दौरान छाया पर विचार करें।"
    },
    'es': { # Spanish
        'no_data': "No hay datos del sensor disponibles para proporcionar asesoramiento.",
        'npk_low': "🌱 **{nutrient} bajo ({value:.1f})**: {message}",
        'npk_high': "🌱 **{nutrient} alto ({value:.1f})**: {message}",
        'soil_moisture_low': "💧 **Humedad del suelo baja ({sm:.1f}%)**: {message}",
        'soil_moisture_high': "💧 **Humedad del suelo alta ({sm:.1f}%)**: {message}",
        'temp_low': "🌡️ **Temperatura baja ({temp:.1f}°C)**: {message}",
        'temp_high': "🌡️ **Temperatura alta ({temp:.1f}°C)**: {message}",
        'humidity_low': "💨 **Humedad baja ({hum:.1f}%)**: {message}",
        'humidity_high': "💨 **Humedad alta ({hum:.1f}%)**: {message}",
        'ph_low': "🧪 **pH bajo ({ph_val:.1f})**: {message}",
        'ph_high': "🧪 **pH alto ({ph_val:.1f})**: {message}",
        'ph_off': "🧪 **pH incorrecto ({ph_val:.1f})**: {message}",
        'light_low': "☀️ **Intensidad de luz baja ({light:.1f} lux)**: {message}",
        'light_high': "☀️ **Intensidad de luz alta ({light:.1f} lux)**: {message}",
        'rainfall_low_msg': "🌧️ **Precipitación baja ({rain:.1f} mm)**: {message}",
        'rainfall_high_msg': "🌧️ **Precipitación alta ({rain:.1f} mm)**: {message}",
        'all_good': "✅ ¡Todos los parámetros principales se ven bien! Siga monitoreando regularmente para un crecimiento óptimo.",
        'npk_n_low': "Considere aplicar fertilizante rico en nitrógeno.",
        'npk_n_high': "El exceso de nitrógeno puede promover el crecimiento foliar sobre el desarrollo de frutos/flores.",
        'npk_p_low': "Considere aplicar fertilizante de fósforo para el desarrollo de la raíz.",
        'npk_p_high': "El fósforo alto puede bloquear otros nutrientes.",
        'npk_k_low': "Considere aplicar fertilizante de potasio para la salud general de la planta y la calidad de la fruta.",
        'npk_k_high': "El exceso de potasio puede interferir con la absorción de calcio y magnesio.",
        'wheat_sm_low': "Riegue ligeramente – el trigo necesita 35–50% de humedad del suelo.",
        'rice_sm_low': "El arroz necesita mucha humedad. Asegure un riego adecuado.",
        'maize_sm_low': "El maíz necesita niveles moderados de humedad del suelo.",
        'banana_sm_low': "Mantenga el suelo constantemente húmedo para el plátano.",
        'mango_sm_high': "Evite el encharcamiento. El mango necesita un suelo bien drenado.",
        'grapes_sm_high': "Las uvas prefieren un suelo más seco – evite el riego excesivo.",
        'cotton_sm_low': "El algodón requiere humedad moderada durante la floración.",
        'millet_sorghum_sm_low': "Estos son cultivos resistentes a la sequía pero aún necesitan humedad mínima.",
        'jute_sm_low': "El yute requiere mucha humedad durante el crecimiento.",
        'pomegranate_sm_high': "Evite el riego excesivo de la granada.",
        'melon_sm_low': "Los melones necesitan riego constante, especialmente durante la fructificación.",
        'coconut_sm_low': "Las palmas de coco necesitan altos niveles de humedad.",
        'mothbeans_sm_low': "Las judías polilla son tolerantes a la sequía pero necesitan riego mínimo durante la floración.",
        'mungbean_sm_low': "Asegure un riego regular durante la floración y la formación de vainas.",
        'blackgram_sm_low': "Mantenga una humedad moderada especialmente durante la floración.",
        'lentil_sm_low': "Las lentejas necesitan humedad baja a moderada.",
        'general_sm_low': "Consejo general: Considere el riego para prevenir el estrés por sequía.",
        'general_sm_high': "Consejo general: Asegure un buen drenaje para prevenir el encharcamiento.",
        'wheat_temp_high': "Proporcione sombra o riegue por la noche – la temperatura es demasiado alta para el trigo.",
        'rice_temp_high': "Demasiado calor para el arroz. Considere el riego nocturno o la sombra.",
        'maize_temp_low': "El maíz prefiere el clima cálido (20–30°C).",
        'banana_temp_low': "El plátano es sensible al frío – asegure condiciones cálidas.",
        'mango_temp_low': "El mango requiere temperaturas más cálidas (>20°C).",
        'cotton_temp_low': "El algodón prospera en temperaturas cálidas.",
        'millet_sorghum_temp_low': "El clima cálido es ideal para el mijo/sorgo.",
        'coffee_temp_low': "El café prospera en el rango de 18–24°C.",
        'jute_temp_low': "El yute crece bien a 25–30°C.",
        'papaya_temp_low': "La papaya prefiere el rango de 21–33°C.",
        'pomegranate_temp_low': "La temperatura ideal es superior a 20°C.",
        'melon_temp_low': "Asegure que la temperatura sea cálida (>25°C).",
        'coconut_temp_low': "La temperatura ideal para el coco es superior a 25°C.",
        'mothbeans_temp_low': "La temperatura debe ser superior a 22°C.",
        'mungbean_temp_low': "La judía mungo requiere condiciones cálidas para un crecimiento óptimo.",
        'blackgram_temp_low': "El rango de temperatura ideal es de 25–35°C.",
        'lentil_temp_low': "Las lentejas crecen bien a 18–30°C.",
        'general_temp_low': "Consejo general: Las bajas temperaturas pueden atrofiar el crecimiento. Considere medidas de protección.",
        'general_temp_high': "Consejo general: Las altas temperaturas pueden causar estrés por calor. Asegure agua y sombra adecuadas.",
        'wheat_hum_high': "Tenga cuidado con las infecciones fúngicas – asegure el flujo de aire.",
        'rice_hum_low': "Aumente la humedad ambiental o use mantillo.",
        'banana_hum_low': "El plátano requiere alta humedad. Considere la nebulización o el acolchado.",
        'grapes_hum_high': "La alta humedad puede provocar infecciones fúngicas.",
        'coffee_hum_low': "El café prefiere alta humedad.",
        'orange_hum_high': "Pode los árboles para mejorar el flujo de aire y prevenir problemas fúngicos.",
        'general_hum_low': "Consejo general: La baja humedad puede causar marchitamiento. Considere la nebulización o el aumento de la humedad del suelo.",
        'general_hum_high': "Consejo general: La alta humedad aumenta el riesgo de enfermedades fúngicas. Asegure una buena ventilación.",
        'wheat_ph_low': "Ligeramente ácido – considere aplicar cal para aumentar el pH.",
        'rice_ph_off': "Mantenga el suelo ligeramente ácido para el arroz (pH 5.5–6.5).",
        'maize_ph_off': "Mantenga el pH del suelo entre 5.8–7.0.",
        'papaya_ph_low': "El suelo ligeramente ácido a neutro es el mejor para la papaya.",
        'orange_ph_off': "El pH ideal del suelo para la naranja es 6.0–7.5.",
        'general_ph_very_low': "Consejo general: El suelo es demasiado ácido. Aplique cal para aumentar el pH y mejorar la disponibilidad de nutrientes.",
        'general_ph_very_high': "Consejo general: El suelo es demasiado alcalino. Aplique azufre o materia orgánica para disminuir el pH.",
        'general_ph_off': "Consejo general: El rango de pH óptimo para la mayoría de los cultivos es 5.5-7.5. Ajuste según sea necesario.",
        'wheat_light_low': "Asegure que el cultivo reciba suficiente luz solar.",
        'rice_light_low': "Asegure que el arroz reciba plena exposición al sol.",
        'general_light_low': "Consejo general: La luz insuficiente puede dificultar la fotosíntesis. Considere la iluminación suplementaria o la poda.",
        'general_light_high': "Consejo general: La luz excesiva puede causar quemaduras. Considere la sombra durante las horas pico."
    },
    'fr': { # French
        'no_data': "Aucune donnée de capteur disponible pour fournir des conseils.",
        'npk_low': "🌱 **{nutrient} est faible ({value:.1f})**: {message}",
        'npk_high': "🌱 **{nutrient} est élevé ({value:.1f})**: {message}",
        'soil_moisture_low': "💧 **Humidité du sol faible ({sm:.1f}%)**: {message}",
        'soil_moisture_high': "💧 **Humidité du sol élevée ({sm:.1f}%)**: {message}",
        'temp_low': "🌡️ **Température basse ({temp:.1f}°C)**: {message}",
        'temp_high': "🌡️ **Température élevée ({temp:.1f}°C)**: {message}",
        'humidity_low': "💨 **Humidité faible ({hum:.1f}%)**: {message}",
        'humidity_high': "💨 **Humidité élevée ({hum:.1f}%)**: {message}",
        'ph_low': "🧪 **pH faible ({ph_val:.1f})**: {message}",
        'ph_high': "🧪 **pH élevé ({ph_val:.1f})**: {message}",
        'ph_off': "🧪 **pH incorrect ({ph_val:.1f})**: {message}",
        'light_low': "☀️ **Intensité lumineuse faible ({light:.1f} lux)**: {message}",
        'light_high': "☀️ **Intensité lumineuse élevée ({light:.1f} lux)**: {message}",
        'rainfall_low_msg': "🌧️ **Précipitations faibles ({rain:.1f} mm)**: {message}",
        'rainfall_high_msg': "🌧️ **Précipitations élevées ({rain:.1f} mm)**: {message}",
        'all_good': "✅ Tous les paramètres majeurs semblent bons ! Continuez à surveiller régulièrement pour une croissance optimale.",
        'npk_n_low': "Envisagez d'appliquer un engrais riche en azote.",
        'npk_n_high': "L'excès d'azote peut favoriser la croissance des feuilles au détriment du développement des fruits/fleurs.",
        'npk_p_low': "Envisagez d'appliquer un engrais phosphoré pour le développement des racines.",
        'npk_p_high': "Un niveau élevé de phosphore peut bloquer d'autres nutriments.",
        'npk_k_low': "Envisagez d'appliquer un engrais potassique pour la santé générale des plantes et la qualité des fruits.",
        'npk_k_high': "L'excès de potassium peut interférer avec l'absorption du calcium et du magnésium.",
        'wheat_sm_low': "Arrosez légèrement – le blé a besoin de 35 à 50% d'humidité du sol.",
        'rice_sm_low': "Le riz a besoin de beaucoup d'humidité. Assurez une irrigation adéquate.",
        'maize_sm_low': "Le maïs a besoin de niveaux d'humidité du sol modérés.",
        'banana_sm_low': "Gardez le sol constamment humide pour la banane.",
        'mango_sm_high': "Évitez l'engorgement. La mangue a besoin d'un sol bien drainé.",
        'grapes_sm_high': "Les raisins préfèrent un sol plus sec – évitez le sur-arrosage.",
        'cotton_sm_low': "Le coton nécessite une humidité modérée pendant la floraison.",
        'millet_sorghum_sm_low': "Ce sont des cultures résistantes à la sécheresse mais nécessitent tout de même une humidité minimale.",
        'jute_sm_low': "Le jute nécessite une humidité abondante pendant la croissance.",
        'pomegranate_sm_high': "Évitez de trop arroser la grenade.",
        'melon_sm_low': "Les melons ont besoin d'un arrosage constant, surtout pendant la fructification.",
        'coconut_sm_low': "Les cocotiers ont besoin de niveaux d'humidité élevés.",
        'mothbeans_sm_low': "Les haricots papillons sont tolérants à la sécheresse mais nécessitent une irrigation minimale pendant la floraison.",
        'mungbean_sm_low': "Assurez un arrosage régulier pendant la floraison et la formation des gousses.",
        'blackgram_sm_low': "Maintenez une humidité modérée, surtout pendant la floraison.",
        'lentil_sm_low': "Les lentilles ont besoin d'une humidité faible à modérée.",
        'general_sm_low': "Conseil général : Envisagez l'irrigation pour prévenir le stress hydrique.",
        'general_sm_high': "Conseil général : Assurez un bon drainage pour prévenir l'engorgement.",
        'wheat_temp_high': "Fournissez de l'ombre ou arrosez le soir – la température est trop élevée pour le blé.",
        'rice_temp_high': "Trop chaud pour le riz. Envisagez l'irrigation nocturne ou l'ombre.",
        'maize_temp_low': "Le maïs préfère le temps chaud (20–30°C).",
        'banana_temp_low': "La banane est sensible au froid – assurez des conditions chaudes.",
        'mango_temp_low': "La mangue nécessite des températures plus chaudes (>20°C).",
        'cotton_temp_low': "Le coton prospère sous des températures chaudes.",
        'millet_sorghum_temp_low': "Le climat chaud est idéal pour le millet/sorgho.",
        'coffee_temp_low': "Le café prospère dans la plage de 18–24°C.",
        'jute_temp_low': "Le jute pousse bien entre 25 et 30°C.",
        'papaya_temp_low': "La papaye préfère la plage de 21–33°C.",
        'pomegranate_temp_low': "La température idéale est supérieure à 20°C.",
        'melon_temp_low': "Assurez-vous que la température est chaude (>25°C).",
        'coconut_temp_low': "La température idéale pour la noix de coco est supérieure à 25°C.",
        'mothbeans_temp_low': "La température doit être supérieure à 22°C.",
        'mungbean_temp_low': "Le haricot mungo nécessite des conditions chaudes pour une croissance optimale.",
        'blackgram_temp_low': "La plage de température idéale est de 25–35°C.",
        'lentil_temp_low': "Les lentilles poussent bien entre 18 et 30°C.",
        'general_temp_low': "Conseil général : Les basses températures peuvent retarder la croissance. Envisagez des mesures de protection.",
        'general_temp_high': "Conseil général : Les températures élevées peuvent provoquer un stress thermique. Assurez un apport suffisant en eau et en ombre.",
        'wheat_hum_high': "Attention aux infections fongiques – assurez une bonne circulation de l'air.",
        'rice_hum_low': "Augmentez l'humidité ambiante ou utilisez du paillis.",
        'banana_hum_low': "La banane nécessite une humidité élevée. Envisagez la brumisation ou le paillage.",
        'grapes_hum_high': "Une humidité élevée peut entraîner des infections fongiques.",
        'coffee_hum_low': "Le café préfère une humidité élevée.",
        'orange_hum_high': "Taillez les arbres pour améliorer la circulation de l'air et prévenir les problèmes fongiques.",
        'general_hum_low': "Conseil général : Une faible humidité peut provoquer le flétrissement. Envisagez la brumisation ou l'augmentation de l'humidité du sol.",
        'general_hum_high': "Conseil général : Une humidité élevée augmente le risque de maladies fongiques. Assurez une bonne ventilation.",
        'wheat_ph_low': "Légèrement acide – envisagez d'appliquer de la chaux pour augmenter le pH.",
        'rice_ph_off': "Maintenez un sol légèrement acide pour le riz (pH 5.5–6.5).",
        'maize_ph_off': "Maintenez le pH du sol entre 5.8 et 7.0.",
        'papaya_ph_low': "Un sol légèrement acide à neutre est le meilleur pour la papaye.",
        'orange_ph_off': "Le pH idéal du sol pour l'orange est de 6.0 à 7.5.",
        'general_ph_very_low': "Conseil général : Le sol est trop acide. Appliquez de la chaux pour augmenter le pH et améliorer la disponibilité des nutriments.",
        'general_ph_very_high': "Conseil général : Le sol est trop alcalin. Appliquez du soufre ou de la matière organique pour diminuer le pH.",
        'general_ph_off': "Conseil général : La plage de pH optimale pour la plupart des cultures est de 5.5 à 7.5. Ajustez si nécessaire.",
        'wheat_light_low': "Assurez-vous que la culture reçoit suffisamment de lumière du soleil.",
        'rice_light_low': "Assurez-vous que le riz reçoit une exposition complète au soleil.",
        'general_light_low': "Conseil général : Une lumière insuffisante peut entraver la photosynthèse. Envisagez un éclairage supplémentaire ou une taille.",
        'general_light_high': "Conseil général : Une lumière excessive peut provoquer des brûlures. Envisagez l'ombrage pendant les heures de pointe."
    },
    'de': { # German
        'no_data': "Keine Sensordaten verfügbar, um Ratschläge zu geben.",
        'npk_low': "🌱 **{nutrient} ist niedrig ({value:.1f})**: {message}",
        'npk_high': "🌱 **{nutrient} ist hoch ({value:.1f})**: {message}",
        'soil_moisture_low': "💧 **Bodenfeuchtigkeit niedrig ({sm:.1f}%)**: {message}",
        'soil_moisture_high': "💧 **Bodenfeuchtigkeit hoch ({sm:.1f}%)**: {message}",
        'temp_low': "🌡️ **Temperatur niedrig ({temp:.1f}°C)**: {message}",
        'temp_high': "🌡️ **Temperatur hoch ({temp:.1f}°C)**: {message}",
        'humidity_low': "💨 **Luftfeuchtigkeit niedrig ({hum:.1f}%)**: {message}",
        'humidity_high': "💨 **Luftfeuchtigkeit hoch ({hum:.1f}%)**: {message}",
        'ph_low': "🧪 **pH-Wert niedrig ({ph_val:.1f})**: {message}",
        'ph_high': "🧪 **pH-Wert hoch ({ph_val:.1f})**: {message}",
        'ph_off': "🧪 **pH-Wert nicht optimal ({ph_val:.1f})**: {message}",
        'light_low': "☀️ **Lichtintensität niedrig ({light:.1f} Lux)**: {message}",
        'light_high': "☀️ **Lichtintensität hoch ({light:.1f} Lux)**: {message}",
        'rainfall_low_msg': "🌧️ **Niederschlag niedrig ({rain:.1f} mm)**: {message}",
        'rainfall_high_msg': "🌧️ **Niederschlag hoch ({rain:.1f} mm)**: {message}",
        'all_good': "✅ Alle wichtigen Parameter sehen gut aus! Überwachen Sie regelmäßig für optimales Wachstum.",
        'npk_n_low': "Erwägen Sie die Anwendung von stickstoffreichem Dünger.",
        'npk_n_high': "Überschüssiger Stickstoff kann das Blattwachstum gegenüber der Frucht-/Blütenentwicklung fördern.",
        'npk_p_low': "Erwägen Sie die Anwendung von Phosphordünger für die Wurzelentwicklung.",
        'npk_p_high': "Hoher Phosphor kann andere Nährstoffe blockieren.",
        'npk_k_low': "Erwägen Sie die Anwendung von Kaliumdünger für die allgemeine Pflanzengesundheit und Fruchtqualität.",
        'npk_k_high': "Überschüssiges Kalium kann die Aufnahme von Kalzium und Magnesium beeinträchtigen.",
        'wheat_sm_low': "Leicht bewässern – Weizen benötigt 35–50% Bodenfeuchtigkeit.",
        'rice_sm_low': "Reis benötigt hohe Feuchtigkeit. Sorgen Sie für eine ordnungsgemäße Bewässerung.",
        'maize_sm_low': "Mais benötigt moderate Bodenfeuchtigkeitswerte.",
        'banana_sm_low': "Halten Sie den Boden für Bananen stets feucht.",
        'mango_sm_high': "Vermeiden Sie Staunässe. Mangos benötigen gut durchlässigen Boden.",
        'grapes_sm_high': "Trauben bevorzugen trockeneren Boden – vermeiden Sie Überwässerung.",
        'cotton_sm_low': "Baumwolle benötigt während der Blütezeit moderate Feuchtigkeit.",
        'millet_sorghum_sm_low': "Dies sind trockenheitstolerante Kulturen, benötigen aber dennoch minimale Feuchtigkeit.",
        'jute_sm_low': "Jute benötigt während des Wachstums reichlich Feuchtigkeit.",
        'pomegranate_sm_high': "Vermeiden Sie Überwässerung bei Granatäpfeln.",
        'melon_sm_low': "Melonen benötigen konstante Bewässerung, besonders während der Fruchtbildung.",
        'coconut_sm_low': "Kokospalmen benötigen hohe Feuchtigkeitswerte.",
        'mothbeans_sm_low': "Mothbohnen sind trockenheitstolerant, benötigen aber während der Blütezeit minimale Bewässerung.",
        'mungbean_sm_low': "Sorgen Sie für regelmäßige Bewässerung während der Blüte und Hülsenbildung.",
        'blackgram_sm_low': "Halten Sie die Feuchtigkeit besonders während der Blüte moderat.",
        'lentil_sm_low': "Linsen benötigen geringe bis moderate Feuchtigkeit.",
        'general_sm_low': "Allgemeiner Ratschlag: Erwägen Sie Bewässerung, um Trockenstress vorzubeugen.",
        'general_sm_high': "Allgemeiner Ratschlag: Sorgen Sie für eine gute Drainage, um Staunässe zu vermeiden.",
        'wheat_temp_high': "Schatten spenden oder abends bewässern – Temperatur ist zu hoch für Weizen.",
        'rice_temp_high': "Zu heiß für Reis. Erwägen Sie abendliche Bewässerung oder Schatten.",
        'maize_temp_low': "Mais bevorzugt warmes Wetter (20–30°C).",
        'banana_temp_low': "Banane ist kälteempfindlich – sorgen Sie für warme Bedingungen.",
        'mango_temp_low': "Mango benötigt wärmere Temperaturen (>20°C).",
        'cotton_temp_low': "Baumwolle gedeiht bei warmen Temperaturen.",
        'millet_sorghum_temp_low': "Warmes Klima ist ideal für Hirse/Sorghum.",
        'coffee_temp_low': "Kaffee gedeiht im Bereich von 18–24°C.",
        'jute_temp_low': "Jute wächst gut bei 25–30°C.",
        'papaya_temp_low': "Papaya bevorzugt den Bereich von 21–33°C.",
        'pomegranate_temp_low': "Ideale Temperatur liegt über 20°C.",
        'melon_temp_low': "Stellen Sie sicher, dass die Temperatur warm ist (>25°C).",
        'coconut_temp_low': "Ideale Temperatur für Kokosnuss liegt über 25°C.",
        'mothbeans_temp_low': "Die Temperatur sollte über 22°C liegen.",
        'mungbean_temp_low': "Mungbohnen benötigen warme Bedingungen für optimales Wachstum.",
        'blackgram_temp_low': "Der ideale Temperaturbereich liegt bei 25–35°C.",
        'lentil_temp_low': "Linsen wachsen gut bei 18–30°C.",
        'general_temp_low': "Allgemeiner Ratschlag: Kalte Temperaturen können das Wachstum hemmen. Erwägen Sie Schutzmaßnahmen.",
        'general_temp_high': "Allgemeiner Ratschlag: Hohe Temperaturen können Hitzestress verursachen. Sorgen Sie für ausreichend Wasser und Schatten.",
        'wheat_hum_high': "Achten Sie auf Pilzinfektionen – sorgen Sie für Luftzirkulation.",
        'rice_hum_low': "Erhöhen Sie die Umgebungsfeuchtigkeit oder verwenden Sie Mulch.",
        'banana_hum_low': "Banane benötigt hohe Luftfeuchtigkeit. Erwägen Sie Besprühen oder Mulchen.",
        'grapes_hum_high': "Hohe Luftfeuchtigkeit kann zu Pilzinfektionen führen.",
        'coffee_hum_low': "Kaffee bevorzugt hohe Luftfeuchtigkeit.",
        'orange_hum_high': "Beschneiden Sie Bäume, um die Luftzirkulation zu verbessern und Pilzprobleme zu vermeiden.",
        'general_hum_low': "Allgemeiner Ratschlag: Geringe Luftfeuchtigkeit kann Welken verursachen. Erwägen Sie Besprühen oder Erhöhung der Bodenfeuchtigkeit.",
        'general_hum_high': "Allgemeiner Ratschlag: Hohe Luftfeuchtigkeit erhöht das Risiko von Pilzkrankheiten. Sorgen Sie für gute Belüftung.",
        'wheat_ph_low': "Leicht sauer – erwägen Sie die Anwendung von Kalk, um den pH-Wert zu erhöhen.",
        'rice_ph_off': "Halten Sie den Boden für Reis leicht sauer (pH 5.5–6.5).",
        'maize_ph_off': "Halten Sie den Boden-pH-Wert zwischen 5.8–7.0.",
        'papaya_ph_low': "Leicht saurer bis neutraler Boden ist am besten für Papaya.",
        'orange_ph_off': "Der ideale Boden-pH-Wert für Orangen liegt bei 6.0–7.5.",
        'general_ph_very_low': "Allgemeiner Ratschlag: Der Boden ist zu sauer. Wenden Sie Kalk an, um den pH-Wert zu erhöhen und die Nährstoffverfügbarkeit zu verbessern.",
        'general_ph_very_high': "Allgemeiner Ratschlag: Der Boden ist zu alkalisch. Wenden Sie Schwefel oder organische Substanz an, um den pH-Wert zu senken.",
        'general_ph_off': "Allgemeiner Ratschlag: Der optimale pH-Bereich für die meisten Kulturen liegt bei 5.5-7.5. Passen Sie ihn bei Bedarf an.",
        'wheat_light_low': "Stellen Sie sicher, dass die Ernte ausreichend Sonnenlicht erhält.",
        'rice_light_low': "Stellen Sie sicher, dass Reis volle Sonneneinstrahlung erhält.",
        'general_light_low': "Allgemeiner Ratschlag: Unzureichendes Licht kann die Photosynthese behindern. Erwägen Sie zusätzliche Beleuchtung oder Beschneidung.",
        'general_light_high': "Allgemeiner Ratschlag: Übermäßiges Licht kann Verbrennungen verursachen. Erwägen Sie Beschattung während der Spitzenzeiten."
    },
    'ar': { # Arabic (Example, requires more detailed translation)
        'no_data': "لا توجد بيانات مستشعر متاحة لتقديم المشورة.",
        'npk_low': "🌱 **{nutrient} منخفض ({value:.1f})**: {message}",
        'npk_high': "🌱 **{nutrient} مرتفع ({value:.1f})**: {message}",
        'soil_moisture_low': "💧 **رطوبة التربة منخفضة ({sm:.1f}%)**: {message}",
        'soil_moisture_high': "💧 **رطوبة التربة مرتفعة ({sm:.1f}%)**: {message}",
        'temp_low': "🌡️ **درجة الحرارة منخفضة ({temp:.1f}°C)**: {message}",
        'temp_high': "🌡️ **درجة الحرارة مرتفعة ({temp:.1f}°C)**: {message}",
        'humidity_low': "💨 **الرطوبة منخفضة ({hum:.1f}%)**: {message}",
        'humidity_high': "💨 **الرطوبة مرتفعة ({hum:.1f}%)**: {message}",
        'ph_low': "🧪 **الرقم الهيدروجيني منخفض ({ph_val:.1f})**: {message}",
        'ph_high': "🧪 **الرقم الهيدروجيني مرتفع ({ph_val:.1f})**: {message}",
        'ph_off': "🧪 **الرقم الهيدروجيني غير صحيح ({ph_val:.1f})**: {message}",
        'light_low': "☀️ **شدة الإضاءة منخفضة ({light:.1f} لوكس)**: {message}",
        'light_high': "☀️ **شدة الإضاءة مرتفعة ({light:.1f} لوكس)**: {message}",
        'rainfall_low_msg': "🌧️ **هطول الأمطار منخفض ({rain:.1f} مم)**: {message}",
        'rainfall_high_msg': "🌧️ **هطول الأمطار مرتفع ({rain:.1f} مم)**: {message}",
        'all_good': "✅ جميع المعلمات الرئيسية تبدو جيدة! استمر في المراقبة بانتظام للنمو الأمثل.",
        'npk_n_low': "فكر في استخدام سماد غني بالنيتروجين.",
        'npk_n_high': "النيتروجين الزائد يمكن أن يعزز نمو الأوراق على حساب نمو الفاكهة/الزهور.",
        'npk_p_low': "فكر في استخدام سماد الفوسفور لتنمية الجذور.",
        'npk_p_high': "الفوسفور العالي يمكن أن يمنع امتصاص العناصر الغذائية الأخرى.",
        'npk_k_low': "فكر في استخدام سماد البوتاسيوم لصحة النبات بشكل عام وجودة الفاكهة.",
        'npk_k_high': "البوتاسيوم الزائد يمكن أن يتداخل مع امتصاص الكالسيوم والمغنيسيوم.",
        'wheat_sm_low': "الري الخفيف – القمح يحتاج إلى 35-50% رطوبة التربة.",
        'rice_sm_low': "الأرز يحتاج إلى رطوبة عالية. تأكد من الري المناسب.",
        'maize_sm_low': "الذرة تحتاج إلى مستويات رطوبة تربة معتدلة.",
        'banana_sm_low': "حافظ على رطوبة التربة باستمرار للموز.",
        'mango_sm_high': "تجنب تشبع التربة بالماء. المانجو يحتاج إلى تربة جيدة التصريف.",
        'grapes_sm_high': "العنب يفضل التربة الأكثر جفافاً – تجنب الإفراط في الري.",
        'cotton_sm_low': "القطن يتطلب رطوبة معتدلة أثناء الإزهار.",
        'millet_sorghum_sm_low': "هذه محاصيل مقاومة للجفاف ولكنها لا تزال بحاجة إلى الحد الأدنى من الرطوبة.",
        'jute_sm_low': "الجوت يتطلب رطوبة وفيرة أثناء النمو.",
        'pomegranate_sm_high': "تجنب الإفراط في ري الرمان.",
        'melon_sm_low': "البطيخ يحتاج إلى ري مستمر، خاصة أثناء الإثمار.",
        'coconut_sm_low': "أشجار النخيل تحتاج إلى مستويات رطوبة عالية.",
        'mothbeans_sm_low': "المحاصيل البقولية مقاومة للجفاف ولكنها تحتاج إلى الحد الأدنى من الري أثناء الإزهار.",
        'mungbean_sm_low': "تأكد من الري المنتظم أثناء الإزهار وتكوين القرون.",
        'blackgram_sm_low': "حافظ على رطوبة معتدلة خاصة أثناء الإزهار.",
        'lentil_sm_low': "العدس ينمو جيدًا في 18-30 درجة مئوية.",
        'general_sm_low': "نصيحة عامة: فكر في الري لمنع إجهاد الجفاف.",
        'general_sm_high': "نصيحة عامة: تأكد من التصريف الجيد لمنع تشبع التربة بالماء.",
        'wheat_temp_high': "وفر الظل أو الري في المساء – درجة الحرارة مرتفعة جدًا للقمح.",
        'rice_temp_high': "ساخن جدًا للأرز. فكر في الري المسائي أو الظل.",
        'maize_temp_low': "الذرة تفضل الطقس الدافئ (20-30 درجة مئوية).",
        'banana_temp_low': "الموز حساس للبرد – تأكد من توفر ظروف دافئة.",
        'mango_temp_low': "المانجو يتطلب درجات حرارة أكثر دفئًا (>20 درجة مئوية).",
        'cotton_temp_low': "القطن يزدهر في درجات الحرارة الدافئة.",
        'millet_sorghum_temp_low': "المناخ الدافئ مثالي للدخن/الذرة الرفيعة.",
        'coffee_temp_low': "القهوة تزدهر في نطاق 18-24 درجة مئوية.",
        'jute_temp_low': "الجوت ينمو جيدًا في 25-30 درجة مئوية.",
        'papaya_temp_low': "البابايا تفضل نطاق 21-33 درجة مئوية.",
        'pomegranate_temp_low': "درجة الحرارة المثالية أعلى من 20 درجة مئوية.",
        'melon_temp_low': "تأكد من أن درجة الحرارة دافئة (>25 درجة مئوية).",
        'coconut_temp_low': "درجة الحرارة المثالية لجوز الهند أعلى من 25 درجة مئوية.",
        'mothbeans_temp_low': "يجب أن تكون درجة الحرارة أعلى من 22 درجة مئوية.",
        'mungbean_temp_low': "المحاصيل البقولية تحتاج إلى ظروف دافئة للنمو الأمثل.",
        'blackgram_temp_low': "نطاق درجة الحرارة المثالي هو 25-35 درجة مئوية.",
        'lentil_temp_low': "العدس ينمو جيدًا في 18-30 درجة مئوية.",
        'general_temp_low': "نصيحة عامة: درجات الحرارة المنخفضة يمكن أن تعيق النمو. فكر في تدابير وقائية.",
        'general_temp_high': "نصيحة عامة: درجات الحرارة المرتفعة يمكن أن تسبب إجهادًا حراريًا. تأكد من توفر الماء والظل الكافيين.",
        'wheat_hum_high': "احذر من الالتهابات الفطرية – تأكد من تدفق الهواء.",
        'rice_hum_low': "زيادة الرطوبة المحيطة أو استخدام النشارة.",
        'banana_hum_low': "الموز يتطلب رطوبة عالية. فكر في الرش أو التغطية بالنشارة.",
        'grapes_hum_high': "الرطوبة العالية قد تؤدي إلى التهابات فطرية.",
        'coffee_hum_low': "القهوة تفضل الرطوبة العالية.",
        'orange_hum_high': "تقليم الأشجار لتحسين تدفق الهواء ومنع مشاكل الفطريات.",
        'general_hum_low': "نصيحة عامة: الرطوبة المنخفضة يمكن أن تسبب الذبول. فكر في الرش أو زيادة رطوبة التربة.",
        'general_hum_high': "نصيحة عامة: الرطوبة العالية تزيد من خطر الأمراض الفطرية. تأكد من التهوية الجيدة.",
        'wheat_ph_low': "حمضي قليلاً – فكر في استخدام الجير لرفع الرقم الهيدروجيني.",
        'rice_ph_off': "حافظ على تربة حمضية قليلاً للأرز (الرقم الهيدروجيني 5.5-6.5).",
        'maize_ph_off': "حافظ على الرقم الهيدروجيني للتربة بين 5.8-7.0.",
        'papaya_ph_low': "التربة الحمضية قليلاً إلى المحايدة هي الأفضل للبابايا.",
        'orange_ph_off': "الرقم الهيدروجيني المثالي للتربة للبرتقال هو 6.0-7.5.",
        'general_ph_very_low': "نصيحة عامة: التربة شديدة الحموضة. استخدم الجير لزيادة الرقم الهيدروجيني وتحسين توافر المغذيات.",
        'general_ph_very_high': "نصيحة عامة: التربة شديدة القلوية. استخدم الكبريت أو المواد العضوية لتقليل الرقم الهيدروجيني.",
        'general_ph_off': "نصيحة عامة: نطاق الرقم الهيدروجيني الأمثل لمعظم المحاصيل هو 5.5-7.5. اضبط حسب الحاجة.",
        'wheat_light_low': "تأكد من حصول المحصول على ما يكفي من ضوء الشمس.",
        'rice_light_low': "تأكد من حصول الأرز على التعرض الكامل لأشعة الشمس.",
        'general_light_low': "نصيحة عامة: الضوء غير الكافي يمكن أن يعيق التمثيل الضوئي. فكر في الإضاءة التكميلية أو التقليم.",
        'general_light_high': "نصيحة عامة: الضوء الزائد يمكن أن يسبب حروقًا. فكر في التظليل خلال ساعات الذروة."
    },
    'ja': { # Japanese (Example)
        'no_data': "アドバイスを提供するためのセンサーデータがありません。",
        'npk_low': "🌱 **{nutrient}が低い ({value:.1f})**: {message}",
        'npk_high': "🌱 **{nutrient}が高い ({value:.1f})**: {message}",
        'soil_moisture_low': "💧 **土壌水分が低い ({sm:.1f}%)**: {message}",
        'soil_moisture_high': "💧 **土壌水分が高い ({sm:.1f}%)**: {message}",
        'temp_low': "🌡️ **温度が低い ({temp:.1f}°C)**: {message}",
        'temp_high': "🌡️ **温度が高い ({temp:.1f}°C)**: {message}",
        'humidity_low': "💨 **湿度が低い ({hum:.1f}%)**: {message}",
        'humidity_high': "💨 **湿度が高い ({hum:.1f}%)**: {message}",
        'ph_low': "🧪 **pHが低い ({ph_val:.1f})**: {message}",
        'ph_high': "🧪 **pHが高い ({ph_val:.1f})**: {message}",
        'ph_off': "🧪 **pHが適切ではありません ({ph_val:.1f})**: {message}",
        'light_low': "☀️ **光強度が低い ({light:.1f} ルクス)**: {message}",
        'light_high': "☀️ **光強度が高い ({light:.1f} ルクス)**: {message}",
        'rainfall_low_msg': "🌧️ **降水量が少ない ({rain:.1f} mm)**: {message}",
        'rainfall_high_msg': "🌧️ **降水量が多い ({rain:.1f} mm)**: {message}",
        'all_good': "✅ すべての主要なパラメーターは良好です！最適な成長のために定期的に監視を続けてください。",
        'npk_n_low': "窒素が豊富な肥料の施用を検討してください。",
        'npk_n_high': "過剰な窒素は、果実/花の成長よりも葉の成長を促進する可能性があります。",
        'npk_p_low': "根の発育のためにリン酸肥料の施用を検討してください。",
        'npk_p_high': "リン酸が高いと他の栄養素が吸収されにくくなることがあります。",
        'npk_k_low': "植物全体の健康と果実の品質のためにカリウム肥料の施用を検討してください。",
        'npk_k_high': "過剰なカリウムは、カルシウムとマグネシウムの吸収を妨げる可能性があります。",
        'wheat_sm_low': "軽く灌漑してください – 小麦は35-50%の土壌水分が必要です。",
        'rice_sm_low': "イネは高い水分が必要です。適切な灌漑を確保してください。",
        'maize_sm_low': "トウモロコシは中程度の土壌水分レベルが必要です。",
        'banana_sm_low': "バナナには土壌を常に湿らせておいてください。",
        'mango_sm_high': "水浸しを避けてください。マンゴーは水はけの良い土壌が必要です。",
        'grapes_sm_high': "ブドウは乾燥した土壌を好みます – 水のやりすぎを避けてください。",
        'cotton_sm_low': "綿は開花中に中程度の水分が必要です。",
        'millet_sorghum_sm_low': "これらは干ばつに強い作物ですが、それでも最小限の水分が必要です。",
        'jute_sm_low': "ジュートは成長中に十分な水分が必要です。",
        'pomegranate_sm_high': "ザクロの水のやりすぎを避けてください。",
        'melon_sm_low': "メロンは、特に結実中に継続的な水やりが必要です。",
        'coconut_sm_low': "ココヤシは高い水分レベルが必要です。",
        'mothbeans_sm_low': "モース豆は干ばつに強いですが、開花中に最小限の灌漑が必要です。",
        'mungbean_sm_low': "開花および莢形成中に定期的な灌漑を確保してください。",
        'blackgram_sm_low': "特に開花中に中程度の水分を維持してください。",
        'lentil_sm_low': "レンズ豆は低から中程度の水分が必要です。",
        'general_sm_low': "一般的なアドバイス：干ばつストレスを防ぐために灌漑を検討してください。",
        'general_sm_high': "一般的なアドバイス：水浸しを防ぐために良好な排水を確保してください。",
        'wheat_temp_high': "日陰を提供するか、夕方に灌漑してください – 小麦には温度が高すぎます。",
        'rice_temp_high': "イネには暑すぎます。夕方の灌漑または日陰を検討してください。",
        'maize_temp_low': "トウモロコシは暖かい気候（20-30°C）を好みます。",
        'banana_temp_low': "バナナは寒さに敏感です – 暖かい条件を確保してください。",
        'mango_temp_low': "マンゴーはより暖かい温度（>20°C）が必要です。",
        'cotton_temp_low': "綿は暖かい温度で生育します。",
        'millet_sorghum_temp_low': "暖かい気候はキビ/ソルガムに理想的です。",
        'coffee_temp_low': "コーヒーは18-24°Cの範囲で生育します。",
        'jute_temp_low': "ジュートは25-30°Cでよく育ちます。",
        'papaya_temp_low': "パパイヤは21-33°Cの範囲を好みます。",
        'pomegranate_temp_low': "理想的な温度は20°C以上です。",
        'melon_temp_low': "温度が暖かい（>25°C）ことを確認してください。",
        'coconut_temp_low': "ココナッツの理想的な温度は25°C以上です。",
        'mothbeans_temp_low': "温度は22°C以上である必要があります。",
        'mungbean_temp_low': "緑豆は最適な成長のために暖かい条件が必要です。",
        'blackgram_temp_low': "理想的な温度範囲は25-35°Cです。",
        'lentil_temp_low': "レンズ豆は18-30°Cでよく育ちます。",
        'general_temp_low': "一般的なアドバイス：低温は成長を妨げる可能性があります。保護対策を検討してください。",
        'general_temp_high': "一般的なアドバイス：高温は熱ストレスを引き起こす可能性があります。十分な水と日陰を確保してください。",
        'wheat_hum_high': "真菌感染症に注意してください – 空気循環を確保してください。",
        'rice_hum_low': "周囲の湿度を上げるか、マルチを使用してください。",
        'banana_hum_low': "バナナは高い湿度が必要です。ミストまたはマルチングを検討してください。",
        'grapes_hum_high': "高湿度は真菌感染症につながる可能性があります。",
        'coffee_hum_low': "コーヒーは高い湿度を好みます。",
        'orange_hum_high': "空気循環を改善し、真菌の問題を防ぐために木を剪定してください。",
        'general_hum_low': "一般的なアドバイス：低湿度はしおれを引き起こす可能性があります。ミストまたは土壌水分の増加を検討してください。",
        'general_hum_high': "一般的なアドバイス：高湿度は真菌性疾患のリスクを高めます。換気を良くしてください。",
        'wheat_ph_low': "わずかに酸性 – pHを上げるために石灰の施用を検討してください。",
        'rice_ph_off': "イネにはわずかに酸性の土壌を維持してください（pH 5.5-6.5）。",
        'maize_ph_off': "土壌pHを5.8-7.0の間に維持してください。",
        'papaya_ph_low': "パパイヤにはわずかに酸性から中性の土壌が最適です。",
        'orange_ph_off': "オレンジの理想的な土壌pHは6.0-7.5です。",
        'general_ph_very_low': "一般的なアドバイス：土壌が酸性すぎます。pHを上げ、栄養素の利用可能性を改善するために石灰を施用してください。",
        'general_ph_very_high': "一般的なアドバイス：土壌がアルカリ性すぎます。pHを下げるために硫黄または有機物を施用してください。",
        'general_ph_off': "一般的なアドバイス：ほとんどの作物にとって最適なpH範囲は5.5-7.5です。必要に応じて調整してください。",
        'wheat_light_low': "作物が十分な日光を浴びるようにしてください。",
        'rice_light_low': "イネが十分な日照を浴びるようにしてください。",
        'general_light_low': "一般的なアドバイス：光が不足すると光合成が妨げられる可能性があります。補助照明または剪定を検討してください。",
        'general_light_high': "一般的なアドバイス：過剰な光は焼けを引き起こす可能性があります。ピーク時間帯は日陰を検討してください。"
    },
    'bn': { # Bengali
        'no_data': "পরামর্শ দেওয়ার জন্য কোনো সেন্সর ডেটা উপলব্ধ নেই।",
        'npk_low': "🌱 **{nutrient} কম আছে ({value:.1f})**: {message}",
        'npk_high': "🌱 **{nutrient} বেশি আছে ({value:.1f})**: {message}",
        'soil_moisture_low': "💧 **মাটির আর্দ্রতা কম ({sm:.1f}%)**: {message}",
        'soil_moisture_high': "💧 **মাটির আর্দ্রতা বেশি ({sm:.1f}%)**: {message}",
        'temp_low': "🌡️ **তাপমাত্রা কম ({temp:.1f}°C)**: {message}",
        'temp_high': "🌡️ **তাপমাত্রা বেশি ({temp:.1f}°C)**: {message}",
        'humidity_low': "💨 **আর্দ্রতা কম ({hum:.1f}%)**: {message}",
        'humidity_high': "💨 **আর্দ্রতা বেশি ({hum:.1f}%)**: {message}",
        'ph_low': "🧪 **pH কম ({ph_val:.1f})**: {message}",
        'ph_high': "🧪 **pH বেশি ({ph_val:.1f})**: {message}",
        'ph_off': "🧪 **pH সঠিক নয় ({ph_val:.1f})**: {message}",
        'light_low': "☀️ **আলোর তীব্রতা কম ({light:.1f} lux)**: {message}",
        'light_high': "☀️ **আলোর তীব্রতা বেশি ({light:.1f} lux)**: {message}",
        'rainfall_low_msg': "🌧️ **বৃষ্টিপাত কম ({rain:.1f} মিমি)**: {message}",
        'rainfall_high_msg': "🌧️ **বৃষ্টিপাত বেশি ({rain:.1f} মিমি)**: {message}",
        'all_good': "✅ সমস্ত প্রধান পরামিতি ভালো দেখাচ্ছে! সর্বোত্তম বৃদ্ধির জন্য নিয়মিত পর্যবেক্ষণ চালিয়ে যান।",
        'npk_n_low': "নাইট্রোজেন সমৃদ্ধ সার প্রয়োগের কথা বিবেচনা করুন।",
        'npk_n_high': "অতিরিক্ত নাইট্রোজেন ফল/ফুলের বিকাশের চেয়ে পাতার বৃদ্ধিকে উৎসাহিত করতে পারে।",
        'npk_p_low': "মূল বিকাশের জন্য ফসফরাস সার প্রয়োগের কথা বিবেচনা করুন।",
        'npk_p_high': "উচ্চ ফসফরাস অন্যান্য পুষ্টি উপাদানকে আবদ্ধ করতে পারে।",
        'npk_k_low': "সামগ্রিক গাছের স্বাস্থ্য এবং ফলের গুণমানের জন্য পটাশিয়াম সার প্রয়োগের কথা বিবেচনা করুন।",
        'npk_k_high': "অতিরিক্ত পটাশিয়াম ক্যালসিয়াম এবং ম্যাগনেসিয়ামের শোষণে হস্তক্ষেপ করতে পারে।",
        'wheat_sm_low': "হালকা সেচ দিন – গমের জন্য ৩৫-৫০% মাটির আর্দ্রতা প্রয়োজন।",
        'rice_sm_low': "ধানের জন্য উচ্চ আর্দ্রতা প্রয়োজন। সঠিক সেচ নিশ্চিত করুন।",
        'maize_sm_low': "ভূট্টার জন্য মাঝারি মাটির আর্দ্রতা স্তর প্রয়োজন।",
        'banana_sm_low': "কলার জন্য মাটি consistently moist রাখুন।",
        'mango_sm_high': "জল জমে যাওয়া এড়িয়ে চলুন। আমের জন্য ভালো নিষ্কাশনযুক্ত মাটি প্রয়োজন।",
        'grapes_sm_high': "আঙ্গুর শুষ্ক মাটি পছন্দ করে – অতিরিক্ত জল দেওয়া এড়িয়ে চলুন।",
        'cotton_sm_low': "তুলা ফুল ফোটার সময় মাঝারি আর্দ্রতা প্রয়োজন।",
        'millet_sorghum_sm_low': "এগুলি খরা-প্রতিরোধী ফসল তবে ন্যূনতম আর্দ্রতা প্রয়োজন।",
        'jute_sm_low': "পাটের বৃদ্ধির সময় প্রচুর আর্দ্রতা প্রয়োজন।",
        'pomegranate_sm_high': "ডালিমের অতিরিক্ত জল দেওয়া এড়িয়ে চলুন।",
        'melon_sm_low': "তরমুজের জন্য নিয়মিত জল দেওয়া প্রয়োজন, বিশেষ করে ফল ধরার সময়।",
        'coconut_sm_low': "নারকেল গাছের জন্য উচ্চ আর্দ্রতা স্তর প্রয়োজন।",
        'mothbeans_sm_low': "মোথবীন খরা-সহনশীল তবে ফুল ফোটার সময় ন্যূনতম সেচ প্রয়োজন।",
        'mungbean_sm_low': "ফুল ফোটা এবং শুঁটি গঠনের সময় নিয়মিত সেচ নিশ্চিত করুন।",
        'blackgram_sm_low': "বিশেষ করে ফুল ফোটার সময় মাঝারি আর্দ্রতা বজায় রাখুন।",
        'lentil_sm_low': "মসুরের জন্য কম থেকে মাঝারি আর্দ্রতা প্রয়োজন।",
        'general_sm_low': "সাধারণ পরামর্শ: খরা চাপ প্রতিরোধের জন্য সেচ বিবেচনা করুন।",
        'general_sm_high': "সাধারণ পরামর্শ: জল জমে যাওয়া প্রতিরোধের জন্য ভালো নিষ্কাশন নিশ্চিত করুন।",
        'wheat_temp_high': "ছায়া প্রদান করুন বা সন্ধ্যায় সেচ দিন – গমের জন্য তাপমাত্রা খুব বেশি।",
        'rice_temp_high': "ধানের জন্য খুব গরম। সন্ধ্যায় সেচ বা ছায়া বিবেচনা করুন।",
        'maize_temp_low': "ভূট্টা উষ্ণ আবহাওয়া (২০-৩০°C) পছন্দ করে।",
        'banana_temp_low': "কলা ঠান্ডার প্রতি সংবেদনশীল – উষ্ণ অবস্থা নিশ্চিত করুন।",
        'mango_temp_low': "আমের জন্য উষ্ণ তাপমাত্রা (>২০°C) প্রয়োজন।",
        'cotton_temp_low': "তুলা উষ্ণ তাপমাত্রায় ভালো জন্মায়।",
        'millet_sorghum_temp_low': "উষ্ণ জলবায়ু বাজরা/জোয়ারের জন্য আদর্শ।",
        'coffee_temp_low': "কফি ১৮-২৪°C পরিসরে ভালো জন্মায়।",
        'jute_temp_low': "পাট ২৫-৩০°C এ ভালো জন্মায়।",
        'papaya_temp_low': "পেঁপে ২১-৩৩°C পরিসর পছন্দ করে।",
        'pomegranate_temp_low': "আদর্শ তাপমাত্রা ২০°C এর উপরে।",
        'melon_temp_low': "তাপমাত্রা উষ্ণ (>২৫°C) নিশ্চিত করুন।",
        'coconut_temp_low': "নারকেলের জন্য আদর্শ তাপমাত্রা ২৫°C এর উপরে।",
        'mothbeans_temp_low': "তাপমাত্রা ২২°C এর উপরে হওয়া উচিত।",
        'mungbean_temp_low': "মুগ ডালের সর্বোত্তম বৃদ্ধির জন্য উষ্ণ অবস্থার প্রয়োজন।",
        'blackgram_temp_low': "আদর্শ তাপমাত্রা পরিসর ২৫-৩৫°C।",
        'lentil_temp_low': "মসুর ১৮-৩০°C এ ভালো জন্মায়।",
        'general_temp_low': "সাধারণ পরামর্শ: ঠান্ডা তাপমাত্রা বৃদ্ধি ব্যাহত করতে পারে। সুরক্ষামূলক ব্যবস্থা বিবেচনা করুন।",
        'general_temp_high': "সাধারণ পরামর্শ: উচ্চ তাপমাত্রা তাপ চাপ সৃষ্টি করতে পারে। পর্যাপ্ত জল এবং ছায়া নিশ্চিত করুন।",
        'wheat_hum_high': "ছত্রাক সংক্রমণ থেকে সাবধান – বায়ু চলাচল নিশ্চিত করুন।",
        'rice_hum_low': "পরিবেষ্টিত আর্দ্রতা বাড়ান বা মালচ ব্যবহার করুন।",
        'banana_hum_low': "কলা উচ্চ আর্দ্রতা প্রয়োজন। কুয়াশা বা মালচিং বিবেচনা করুন।",
        'grapes_hum_high': "উচ্চ আর্দ্রতা ছত্রাক সংক্রমণের কারণ হতে পারে।",
        'coffee_hum_low': "কফি উচ্চ আর্দ্রতা পছন্দ করে।",
        'orange_hum_high': "বায়ু চলাচল উন্নত করতে এবং ছত্রাকজনিত সমস্যা প্রতিরোধের জন্য গাছ ছাঁটাই করুন।",
        'general_hum_low': "সাধারণ পরামর্শ: কম আর্দ্রতা শুকিয়ে যেতে পারে। কুয়াশা বা মাটির আর্দ্রতা বাড়ানোর কথা বিবেচনা করুন।",
        'general_hum_high': "সাধারণ পরামর্শ: উচ্চ আর্দ্রতা ছত্রাক রোগের ঝুঁকি বাড়ায়। ভালো বায়ুচলাচল নিশ্চিত করুন।",
        'wheat_ph_low': "সামান্য অম্লীয় – pH বাড়ানোর জন্য চুন প্রয়োগের কথা বিবেচনা করুন।",
        'rice_ph_off': "ধানের জন্য সামান্য অম্লীয় মাটি বজায় রাখুন (pH ৫.৫-৬.৫)।",
        'maize_ph_off': "মাটির pH ৫.৮-৭.০ এর মধ্যে বজায় রাখুন।",
        'papaya_ph_low': "পেঁপের জন্য সামান্য অম্লীয় থেকে নিরপেক্ষ মাটি সবচেয়ে ভালো।",
        'orange_ph_off': "কমলার জন্য আদর্শ মাটির pH ৬.০-৭.৫।",
        'general_ph_very_low': "সাধারণ পরামর্শ: মাটি খুব অম্লীয়। pH বাড়াতে এবং পুষ্টির প্রাপ্যতা উন্নত করতে চুন প্রয়োগ করুন।",
        'general_ph_very_high': "সাধারণ পরামর্শ: মাটি খুব ক্ষারীয়। pH কমাতে সালফার বা জৈব পদার্থ প্রয়োগ করুন।",
        'general_ph_off': "সাধারণ পরামর্শ: বেশিরভাগ ফসলের জন্য সর্বোত্তম pH পরিসর ৫.৫-৭.৫। প্রয়োজন অনুযায়ী সামঞ্জস্য করুন।",
        'wheat_light_low': "ফসল পর্যাপ্ত সূর্যালোক পায় তা নিশ্চিত করুন।",
        'rice_light_low': "ধান সম্পূর্ণ সূর্যালোক পায় তা নিশ্চিত করুন।",
        'general_light_low': "সাধারণ পরামর্শ: অপর্যাপ্ত আলো সালোকসংশ্লেষণকে বাধা দিতে পারে। অতিরিক্ত আলো বা ছাঁটাই বিবেচনা করুন।",
        'general_light_high': "সাধারণ পরামর্শ: অতিরিক্ত আলো ঝলসে যেতে পারে। পিক আওয়ারে ছায়া দেওয়ার কথা বিবেচনা করুন।"
    }
}

def crop_care_advice(df, crop_type, lang='en'):
    """Provides crop-specific care advice based on latest sensor readings."""
    messages = ADVICE_MESSAGES.get(lang, ADVICE_MESSAGES['en']) # Fallback to English

    if df.empty:
        return [messages['no_data']]
    
    latest = df.iloc[-1].to_dict()
    tips = []
    
    ct = crop_type.lower()

    npk_advice = {
        'N': {'min': 50, 'max': 150, 'low_msg': messages['npk_n_low'], 'high_msg': messages['npk_n_high']},
        'P': {'min': 20, 'max': 60, 'low_msg': messages['npk_p_low'], 'high_msg': messages['npk_p_high']},
        'K': {'min': 50, 'max': 200, 'low_msg': messages['npk_k_low'], 'high_msg': messages['npk_k_high']},
    }

    for nutrient, thresholds in npk_advice.items():
        if nutrient in latest and not pd.isna(latest.get(nutrient)):
            value = latest[nutrient]
            if value < thresholds['min']:
                tips.append(messages['npk_low'].format(nutrient=nutrient, value=value, message=thresholds['low_msg']))
            elif value > thresholds['max']:
                tips.append(messages['npk_high'].format(nutrient=nutrient, value=value, message=thresholds['high_msg']))

    # Specific crop advice
    if 'soil_moisture' in latest and not pd.isna(latest.get('soil_moisture')):
        sm = latest['soil_moisture']
        if ct == 'wheat':
            if sm < 35: tips.append(messages['soil_moisture_low'].format(sm=sm, message=messages['wheat_sm_low']))
        elif ct == 'rice':
            if sm < 60: tips.append(messages['soil_moisture_low'].format(sm=sm, message=messages['rice_sm_low']))
        elif ct == 'maize':
            if sm < 40: tips.append(messages['soil_moisture_low'].format(sm=sm, message=messages['maize_sm_low']))
        elif ct == 'banana':
            if sm < 50: tips.append(messages['soil_moisture_low'].format(sm=sm, message=messages['banana_sm_low']))
        elif ct == 'mango':
            if sm > 60: tips.append(messages['soil_moisture_high'].format(sm=sm, message=messages['mango_sm_high']))
        elif ct == 'grapes':
            if sm > 50: tips.append(messages['soil_moisture_high'].format(sm=sm, message=messages['grapes_sm_high']))
        elif ct == 'cotton':
            if sm < 30: tips.append(messages['soil_moisture_low'].format(sm=sm, message=messages['cotton_sm_low']))
        elif ct == 'millet' or ct == 'sorghum':
            if sm < 25: tips.append(messages['soil_moisture_low'].format(sm=sm, message=messages['millet_sorghum_sm_low']))
        elif ct == 'jute':
            if sm < 50: tips.append(messages['soil_moisture_low'].format(sm=sm, message=messages['jute_sm_low']))
        elif ct == 'pomegranate':
            if sm > 50: tips.append(messages['soil_moisture_high'].format(sm=sm, message=messages['pomegranate_sm_high']))
        elif ct == 'muskmelon' or ct == 'watermelon':
            if sm < 30: tips.append(messages['soil_moisture_low'].format(sm=sm, message=messages['melon_sm_low']))
        elif ct == 'coconut':
            if sm < 50: tips.append(messages['soil_moisture_low'].format(sm=sm, message=messages['coconut_sm_low']))
        elif ct == 'mothbeans':
            if sm < 25: tips.append(messages['soil_moisture_low'].format(sm=sm, message=messages['mothbeans_sm_low']))
        elif ct == 'mungbean':
            if sm < 30: tips.append(messages['soil_moisture_low'].format(sm=sm, message=messages['mungbean_sm_low']))
        elif ct == 'blackgram':
            if sm < 35: tips.append(messages['soil_moisture_low'].format(sm=sm, message=messages['blackgram_sm_low']))
        elif ct == 'lentil':
            if sm < 25: tips.append(messages['soil_moisture_low'].format(sm=sm, message=messages['lentil_sm_low']))
        # General advice if not crop-specific
        if sm < 30: tips.append(messages['soil_moisture_low'].format(sm=sm, message=messages['general_sm_low']))
        elif sm > 70: tips.append(messages['soil_moisture_high'].format(sm=sm, message=messages['general_sm_high']))

    if 'temperature' in latest and not pd.isna(latest.get('temperature')):
        temp = latest['temperature']
        if ct == 'wheat':
            if temp > 32: tips.append(messages['temp_high'].format(temp=temp, message=messages['wheat_temp_high']))
        elif ct == 'rice':
            if temp > 38: tips.append(messages['temp_high'].format(temp=temp, message=messages['rice_temp_high']))
        elif ct == 'maize':
            if temp < 20: tips.append(messages['temp_low'].format(temp=temp, message=messages['maize_temp_low']))
        elif ct == 'banana':
            if temp < 15: tips.append(messages['temp_low'].format(temp=temp, message=messages['banana_temp_low']))
        elif ct == 'mango':
            if temp < 20: tips.append(messages['temp_low'].format(temp=temp, message=messages['mango_temp_low']))
        elif ct == 'cotton':
            if temp < 20: tips.append(messages['temp_low'].format(temp=temp, message=messages['cotton_temp_low']))
        elif ct == 'millet' or ct == 'sorghum':
            if temp < 20: tips.append(messages['temp_low'].format(temp=temp, message=messages['millet_sorghum_temp_low']))
        elif ct == 'coffee':
            if temp < 18: tips.append(messages['temp_low'].format(temp=temp, message=messages['coffee_temp_low']))
        elif ct == 'jute':
            if temp < 25: tips.append(messages['temp_low'].format(temp=temp, message=messages['jute_temp_low']))
        elif ct == 'papaya':
            if temp < 20: tips.append(messages['temp_low'].format(temp=temp, message=messages['papaya_temp_low']))
        elif ct == 'pomegranate':
            if temp < 20: tips.append(messages['temp_low'].format(temp=temp, message=messages['pomegranate_temp_low']))
        elif ct == 'muskmelon' or ct == 'watermelon':
            if temp < 25: tips.append(messages['temp_low'].format(temp=temp, message=messages['melon_temp_low']))
        elif ct == 'coconut':
            if temp < 25: tips.append(messages['temp_low'].format(temp=temp, message=messages['coconut_temp_low']))
        elif ct == 'mothbeans':
            if temp < 22: tips.append(messages['temp_low'].format(temp=temp, message=messages['mothbeans_temp_low']))
        elif ct == 'mungbean':
            if temp < 20: tips.append(messages['temp_low'].format(temp=temp, message=messages['mungbean_temp_low']))
        elif ct == 'blackgram':
            if temp < 18: tips.append(messages['temp_low'].format(temp=temp, message=messages['blackgram_temp_low']))
        elif ct == 'lentil':
            if temp < 15: tips.append(messages['temp_low'].format(temp=temp, message=messages['lentil_temp_low']))
        # General advice
        if temp < 18: tips.append(messages['temp_low'].format(temp=temp, message=messages['general_temp_low']))
        elif temp > 35: tips.append(messages['temp_high'].format(temp=temp, message=messages['general_temp_high']))

    if 'humidity' in latest and not pd.isna(latest.get('humidity')):
        hum = latest['humidity']
        if ct == 'wheat':
            if hum > 70: tips.append(messages['humidity_high'].format(hum=hum, message=messages['wheat_hum_high']))
        elif ct == 'rice':
            if hum < 60: tips.append(messages['humidity_low'].format(hum=hum, message=messages['rice_hum_low']))
        elif ct == 'banana':
            if hum < 60: tips.append(messages['humidity_low'].format(hum=hum, message=messages['banana_hum_low']))
        elif ct == 'grapes':
            if hum > 70: tips.append(messages['humidity_high'].format(hum=hum, message=messages['grapes_hum_high']))
        elif ct == 'coffee':
            if hum < 60: tips.append(messages['humidity_low'].format(hum=hum, message=messages['coffee_hum_low']))
        elif ct == 'orange':
            if hum > 70: tips.append(messages['humidity_high'].format(hum=hum, message=messages['orange_hum_high']))
        # General advice
        if hum < 40: tips.append(messages['humidity_low'].format(hum=hum, message=messages['general_hum_low']))
        elif hum > 80: tips.append(messages['humidity_high'].format(hum=hum, message=messages['general_hum_high']))

    # Note: Using 'ph' from fetched data after consistency handling
    if 'ph' in latest and not pd.isna(latest.get('ph')):
        ph_val = latest['ph']
        if ct == 'wheat':
            if ph_val < 6.0: tips.append(messages['ph_low'].format(ph_val=ph_val, message=messages['wheat_ph_low']))
        elif ct == 'rice':
            if ph_val < 5.5 or ph_val > 6.5: tips.append(messages['ph_off'].format(ph_val=ph_val, message=messages['rice_ph_off']))
        elif ct == 'maize':
            if ph_val < 5.8 or ph_val > 7: tips.append(messages['ph_off'].format(ph_val=ph_val, message=messages['maize_ph_off']))
        elif ct == 'papaya':
            if ph_val < 6: tips.append(messages['ph_low'].format(ph_val=ph_val, message=messages['papaya_ph_low']))
        elif ct == 'orange':
            if ph_val < 6 or ph_val > 7.5: tips.append(messages['ph_off'].format(ph_val=ph_val, message=messages['orange_ph_off']))
        # General advice
        if ph_val < 5.5: tips.append(messages['ph_low'].format(ph_val=ph_val, message=messages['general_ph_very_low']))
        elif ph_val > 7.5: tips.append(messages['ph_high'].format(ph_val=ph_val, message=messages['general_ph_very_high']))
        elif not (5.5 <= ph_val <= 7.5): tips.append(messages['ph_off'].format(ph_val=ph_val, message=messages['general_ph_off']))

    if 'light_intensity' in latest and not pd.isna(latest.get('light_intensity')):
        light = latest['light_intensity']
        if ct == 'wheat':
            if light < 400: tips.append(messages['light_low'].format(light=light, message=messages['wheat_light_low']))
        elif ct == 'rice':
            if light < 500: tips.append(messages['light_low'].format(light=light, message=messages['rice_light_low']))
        # General advice
        if light < 300: tips.append(messages['light_low'].format(light=light, message=messages['general_light_low']))
        elif light > 800: tips.append(messages['light_high'].format(light=light, message=messages['general_light_high']))
            
    if 'rainfall' in latest and not pd.isna(latest.get('rainfall')):
        rain = latest['rainfall']
        if rain < 50: 
            tips.append(messages['rainfall_low_msg'].format(rain=rain, message=messages['rainfall_low_msg']))
        elif rain > 200: 
            tips.append(messages['rainfall_high_msg'].format(rain=rain, message=messages['rainfall_high_msg']))
        
    return tips if tips else [messages['all_good']]

# --- Seed Recommender Function ---
# Adding multilingual support for seed recommendations
SEED_RECOMMENDATIONS_MESSAGES = {
    'en': {
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
        'no_specific': "No specific recommendations, as current conditions are unusual or general."
    },
    'hi': {
        'intro': "वर्तमान परिस्थितियों के आधार पर, आप विचार कर सकते हैं: ",
        'outro': ". सटीक सिफारिशों के लिए कृपया स्थानीय कृषि विशेषज्ञों से सलाह लें।",
        'acid_tolerant': "अम्ल-सहिष्णु फसलें जैसे ब्लूबेरी, आलू, या चावल की विशिष्ट किस्में",
        'alkaline_tolerant': "क्षार-सहिष्णु फसलें जैसे शतावरी, पालक, या अल्फाल्फा की विशिष्ट किस्में",
        'neutral_ph': "गेहूं, मक्का, और अधिकांश सब्जियों सहित तटस्थ से थोड़े अम्लीय पीएच (5.5-7.5) में फसलों की एक विस्तृत श्रृंखला पनपती है",
        'heat_tolerant': "गर्मी-सहिष्णु फसलें जैसे बाजरा, ज्वार, कपास, या कुछ प्रकार की फलियां",
        'cold_hardy': "ठंड-सहिष्णु फसलें जैसे गेहूं (शीतकालीन किस्में), जौ, जई, या मटर",
        'warm_season': "गर्मियों की फसलें जैसे मक्का, चावल (उष्णकटिबंधीय), अधिकांश सब्जियां, और फल",
        'drought_resistant': "सूखे प्रतिरोधी फसलें जैसे बाजरा, ज्वार, चना, या कुछ प्रकार की फलियां (जैसे मोठबीन)",
        'water_loving': "पानी पसंद करने वाली फसलें जैसे चावल, गन्ना, जूट, या वे फसलें जो अस्थायी जलभराव को सहन करती हैं",
        'moderate_rainfall': "मध्यम वर्षा के लिए उपयुक्त फसलें, जिनमें गेहूं, मक्का, और कई सब्जियां शामिल हैं",
        'very_dry': "बहुत सूखा-सहिष्णु फसलें (जैसे रेगिस्तान-अनुकूलित तरबूज या कुछ जड़ी-बूटियाँ)",
        'very_wet': "अर्ध-जलीय फसलें या वे जो जलभराव के प्रति अत्यधिक सहिष्णु हैं (जैसे तारो, चावल की कुछ किस्में यदि खराब जल निकासी हो)",
        'no_specific': "कोई विशिष्ट सिफारिश नहीं, क्योंकि वर्तमान परिस्थितियाँ असामान्य या सामान्य हैं।"
    },
    'es': { # Spanish
        'intro': "Basado en las condiciones actuales, podría considerar: ",
        'outro': ". Consulte a expertos agrícolas locales para recomendaciones precisas.",
        'acid_tolerant': "cultivos tolerantes a la acidez como arándanos, patatas o variedades específicas de arroz",
        'alkaline_tolerant': "cultivos tolerantes a la alcalinidad como espárragos, espinacas o variedades específicas de alfalfa",
        'neutral_ph': "una amplia gama de cultivos prosperan en pH neutro a ligeramente ácido (5.5-7.5), incluyendo trigo, maíz y la mayoría de las verduras",
        'heat_tolerant': "cultivos tolerantes al calor como mijo, sorgo, algodón o algunas variedades de frijoles",
        'cold_hardy': "cultivos resistentes al frío como trigo (variedades de invierno), cebada, avena o guisantes",
        'warm_season': "cultivos de estación cálida como maíz, arroz (tropical), la mayoría de las verduras y frutas",
        'drought_resistant': "cultivos resistentes a la sequía como mijo, sorgo, garbanzos o ciertos tipos de frijoles (por ejemplo, frijoles polilla)",
        'water_loving': "cultivos amantes del agua como arroz, caña de azúcar, yute o cultivos que toleran el encharcamiento temporal",
        'moderate_rainfall': "cultivos adecuados para precipitaciones moderadas, incluyendo trigo, maíz y muchas verduras",
        'very_dry': "cultivos muy tolerantes a la sequía (por ejemplo, melones adaptados al desierto o algunas hierbas)",
        'very_wet': "cultivos semiacuáticos o aquellos altamente tolerantes al encharcamiento (por ejemplo, taro, algunas variedades de arroz si están mal drenadas)",
        'no_specific': "No hay recomendaciones específicas, ya que las condiciones actuales son inusuales o generales."
    },
    'fr': { # French
        'intro': "En fonction des conditions actuelles, vous pourriez envisager : ",
        'outro': ". Veuillez consulter des experts agricoles locaux pour des recommandations précises.",
        'acid_tolerant': "cultures tolérantes à l'acidité comme les myrtilles, les pommes de terre ou des variétés spécifiques de riz",
        'alkaline_tolerant': "cultures tolérantes à l'alcalinité telles que les asperges, les épinards ou des variétés spécifiques de luzerne",
        'neutral_ph': "une large gamme de cultures prospèrent dans un pH neutre à légèrement acide (5.5-7.5), y compris le blé, le maïs et la plupart des légumes",
        'heat_tolerant': "cultures tolérantes à la chaleur comme le millet, le sorgho, le coton ou certaines variétés de haricots",
        'cold_hardy': "cultures résistantes au froid comme le blé (variétés d'hiver), l'orge, l'avoine ou les pois",
        'warm_season': "cultures de saison chaude comme le maïs, le riz (tropica), la plupart des légumes et des fruits",
        'drought_resistant': "cultures résistantes à la sécheresse comme le millet, le sorgho, les pois chiches ou certains types de haricots (par exemple, les haricots papillons)",
        'water_loving': "cultures aimant l'eau comme le riz, la canne à sucre, le jute ou les cultures qui tolèrent l'engorgement temporaire",
        'moderate_rainfall': "cultures adaptées aux précipitations modérées, y compris le blé, le maïs et de nombreux légumes",
        'very_dry': "cultures très tolérantes à la sécheresse (par exemple, les melons adaptés au désert ou certaines herbes)",
        'very_wet': "cultures semi-aquatiques ou celles très tolérantes à l'engorgement (par exemple, le taro, certaines variétés de riz si mal drainées)",
        'no_specific': "Aucune recommandation spécifique, car les conditions actuelles sont inhabituelles ou générales."
    },
    'de': { # German
        'intro': "Basierend auf den aktuellen Bedingungen könnten Sie Folgendes in Betracht ziehen: ",
        'outro': ". Bitte konsultieren Sie lokale Landwirtschaftsexperten für präzise Empfehlungen.",
        'acid_tolerant': "säuretolerante Kulturen wie Heidelbeeren, Kartoffeln oder spezifische Reissorten",
        'alkaline_tolerant': "alkalitolerante Kulturen wie Spargel, Spinat oder spezifische Luzernesorten",
        'neutral_ph': "eine breite Palette von Kulturen gedeiht in neutralem bis leicht saurem pH-Wert (5.5-7.5), einschließlich Weizen, Mais und den meisten Gemüsesorten",
        'heat_tolerant': "hitzetolerante Kulturen wie Hirse, Sorghum, Baumwolle oder einige Bohnensorten",
        'cold_hardy': "kälteresistente Kulturen wie Weizen (Winter сорта), Gerste, Hafer oder Erbsen",
        'warm_season': "Warmwetterkulturen wie Mais, Reis (tropisch), die meisten Gemüsesorten und Früchte",
        'drought_resistant': "trockenheitsresistente Kulturen wie Hirse, Sorghum, Kichererbsen oder bestimmte Bohnensorten (z.B. Mothbohnen)",
        'water_loving': "wasserliebende Kulturen wie Reis, Zuckerrohr, Jute oder Kulturen, die vorübergehende Staunässe vertragen",
        'moderate_rainfall': "Kulturen, die für moderate Niederschläge geeignet sind, einschließlich Weizen, Mais und viele Gemüsesorten",
        'very_dry': "sehr trockenheitstolerante Kulturen (z.B. wüstenangepasste Melonen oder einige Kräuter)",
        'very_wet': "semi-aquatische Kulturen oder solche, die sehr tolerant gegenüber Staunässe sind (z.B. Taro, einige Reissorten bei schlechter Drainage)",
        'no_specific': "Keine spezifischen Empfehlungen, da die aktuellen Bedingungen ungewöhnlich oder allgemein sind."
    },
    'ar': { # Arabic (Example, requires more detailed translation)
        'intro': "بناءً على الظروف الحالية، قد تفكر في: ",
        'outro': ". يرجى استشارة خبراء الزراعة المحليين للحصول على توصيات دقيقة.",
        'acid_tolerant': "محاصيل تتحمل الحموضة مثل التوت الأزرق، البطاطس، أو أصناف معينة من الأرز",
        'alkaline_tolerant': "محاصيل تتحمل القلوية مثل الهليون، السبانخ، أو أصناف معينة من البرسيم الحجازي",
        'neutral_ph': "مجموعة واسعة من المحاصيل تزدهر في درجة حموضة محايدة إلى حمضية قليلاً (5.5-7.5)، بما في ذلك القمح والذرة ومعظم الخضروات",
        'heat_tolerant': "محاصيل تتحمل الحرارة مثل الدخن، الذرة الرفيعة، القطن، أو بعض أنواع الفول",
        'cold_hardy': "محاصيل مقاومة للبرد مثل القمح (أصناف الشتاء)، الشعير، الشوفان، أو البازلاء",
        'warm_season': "محاصيل الموسم الدافئ مثل الذرة، الأرز (الاستوائي)، معظم الخضروات، والفواكه",
        'drought_resistant': "محاصيل مقاومة للجفاف مثل الدخن، الذرة الرفيعة، الحمص، أو أنواع معينة من الفول (مثل الماش)",
        'water_loving': "محاصيل محبة للماء مثل الأرز، قصب السكر، الجوت، أو المحاصيل التي تتحمل التشبع بالمياه مؤقتًا",
        'moderate_rainfall': "محاصيل مناسبة للأمطار المعتدلة، بما في ذلك القمح والذرة والعديد من الخضروات",
        'very_dry': "محاصيل شديدة التحمل للجفاف (مثل البطيخ الصحراوي أو بعض الأعشاب)",
        'very_wet': "محاصيل شبه مائية أو تلك شديدة التحمل للتشبع بالمياه (مثل القلقاس، بعض أصناف الأرز إذا كانت التربة سيئة التصريف)",
        'no_specific': "لا توجد توصيات محددة، حيث أن الظروف الحالية غير عادية أو عامة."
    },
    'ja': { # Japanese (Example)
        'intro': "現在の状況に基づき、以下を検討することができます：",
        'outro': "正確な推奨事項については、地元の農業専門家にご相談ください。",
        'acid_tolerant': "ブルーベリー、ジャガイモ、特定のイネ品種などの酸性土壌に強い作物",
        'alkaline_tolerant': "アスパラガス、ほうれん草、特定のアルファルファ品種などのアルカリ性土壌に強い作物",
        'neutral_ph': "小麦、トウモロコシ、ほとんどの野菜など、中性から弱酸性のpH（5.5-7.5）で育つ幅広い作物",
        'heat_tolerant': "キビ、ソルガム、綿、一部の豆類などの耐熱性作物",
        'cold_hardy': "小麦（冬品種）、大麦、オート麦、エンドウ豆などの耐寒性作物",
        'warm_season': "トウモロコシ、イネ（熱帯性）、ほとんどの野菜、果物などの暖季作物",
        'drought_resistant': "キビ、ソルガム、ひよこ豆、特定の種類の豆（例：モス豆）などの干ばつ耐性作物",
        'water_loving': "イネ、サトウキビ、ジュート、一時的な湛水に耐える作物などの水生作物",
        'moderate_rainfall': "小麦、トウモロコシ、多くの野菜など、中程度の降雨に適した作物",
        'very_dry': "非常に干ばつに強い作物（例：砂漠に適応したメロンや一部のハーブ）",
        'very_wet': "半水生作物または湛水に非常に強い作物（例：タロイモ、排水が悪い場合の特定のイネ品種）",
        'no_specific': "現在の状況が異常または一般的なため、特定の推奨事項はありません。"
    },
    'bn': { # Bengali
        'intro': "বর্তমান অবস্থার উপর ভিত্তি করে, আপনি বিবেচনা করতে পারেন: ",
        'outro': ". সঠিক সুপারিশের জন্য স্থানীয় কৃষি বিশেষজ্ঞদের সাথে পরামর্শ করুন।",
        'acid_tolerant': "ব্লুবেরি, আলু, বা নির্দিষ্ট ধান জাতের মতো অ্যাসিড-সহনশীল ফসল",
        'alkaline_tolerant': "শতমূলী, পালং শাক, বা নির্দিষ্ট আলফালফা জাতের মতো ক্ষার-সহনশীল ফসল",
        'neutral_ph': "গম, ভুট্টা এবং বেশিরভাগ সবজি সহ নিরপেক্ষ থেকে সামান্য অম্লীয় pH (৫.৫-৭.৫) এ বিস্তৃত ফসল ভালো জন্মায়",
        'heat_tolerant': "বাজরা, জোয়ার, তুলা, বা কিছু শিম জাতের মতো তাপ-সহনশীল ফসল",
        'cold_hardy': "গম (শীতকালীন জাত), বার্লি, ওটস, বা মটরের মতো ঠান্ডা-সহনশীল ফসল",
        'warm_season': "ভুট্টা, ধান (ক্রান্তীয়), বেশিরভাগ সবজি এবং ফলের মতো উষ্ণ-মৌসুমী ফসল",
        'drought_resistant': "বাজরা, জোয়ার, ছোলা, বা নির্দিষ্ট ধরণের শিম (যেমন মোথবীন) এর মতো খরা-প্রতিরোধী ফসল",
        'water_loving': "ধান, আখ, পাট, বা অস্থায়ী জলজমাট সহনশীল ফসলের মতো জল-প্রেমী ফসল",
        'moderate_rainfall': "গম, ভুট্টা এবং অনেক সবজি সহ মাঝারি বৃষ্টিপাতের জন্য উপযুক্ত ফসল",
        'very_dry': "খুব খরা-সহনশীল ফসল (যেমন মরুভূমি-অভিযোজিত তরমুজ বা কিছু ভেষজ)",
        'very_wet': "আধা-জলজ ফসল বা যেগুলি জলজমাট অত্যন্ত সহনশীল (যেমন কচু, খারাপ নিষ্কাশন হলে কিছু ধান জাত)",
        'no_specific': "কোনো নির্দিষ্ট সুপারিশ নেই, কারণ বর্তমান অবস্থা অস্বাভাবিক বা সাধারণ।"
    }
}

def recommend_seeds(ph, temperature, rainfall, soil_moisture=None, lang='en'):
    """
    Suggests suitable crops based on environmental conditions.
    Args:
        ph (float): Current pH value of the soil.
        temperature (float): Current temperature in Celsius.
        rainfall (float): Recent rainfall in mm.
        soil_moisture (float, optional): Current soil moisture percentage.
        If available, provides more specific advice.
        lang (str): Language for recommendations ('en' for English, 'hi' for Hindi).
    Returns:
        str: Recommended crops or general advice.
    """
    messages = SEED_RECOMMENDATIONS_MESSAGES.get(lang, SEED_RECOMMENDATIONS_MESSAGES['en'])
    recommendations = []

    # pH based recommendations
    if ph < 5.5:
        recommendations.append(messages['acid_tolerant'])
    elif ph > 7.5:
        recommendations.append(messages['alkaline_tolerant'])
    else:
        recommendations.append(messages['neutral_ph'])

    # Temperature based recommendations
    if temperature > 35:
        recommendations.append(messages['heat_tolerant'])
    elif temperature < 15:
        recommendations.append(messages['cold_hardy'])
    else:
        recommendations.append(messages['warm_season'])

    # Rainfall based recommendations
    if rainfall < 50: # Low rainfall
        recommendations.append(messages['drought_resistant'])
    elif rainfall > 200: # High rainfall, potentially waterlogging
        recommendations.append(messages['water_loving'])
    else:
        recommendations.append(messages['moderate_rainfall'])

    # Soil Moisture based recommendations (more granular if available)
    if soil_moisture is not None:
        if soil_moisture < 30: # Very dry
            recommendations.append(messages['very_dry'])
        elif soil_moisture > 80: # Very wet, prone to waterlogging
            recommendations.append(messages['very_wet'])

    if not recommendations:
        return messages['no_specific']
    
    return messages['intro'] + ", ".join(recommendations) + messages['outro']


# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="Smart AgriTech Dashboard", page_icon="🌿")

st.title("🌿 Smart AgriTech Dashboard")

# Refresh button and language selector in a row
col_refresh, col_lang = st.columns([0.15, 0.85])
with col_refresh:
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear() # Clear cache to fetch fresh data
        st.rerun()
with col_lang:
    # Get all available languages from the ADVICE_MESSAGES keys
    available_languages = list(ADVICE_MESSAGES.keys())
    # Create a mapping for display names if needed, e.g., {'en': 'English', 'hi': 'Hindi'}
    language_display_names = {
        'en': 'English',
        'hi': 'Hindi',
        'es': 'Español',
        'fr': 'Français',
        'de': 'Deutsch',
        'ar': 'العربية', # Arabic
        'ja': '日本語', # Japanese
        'bn': 'বাংলা' # Bengali
    }
    # Create a list of display names for the selectbox, maintaining order
    display_options = [language_display_names.get(lang, lang) for lang in available_languages]
    
    # Find the index of 'en' for default selection
    default_lang_index = available_languages.index('en') if 'en' in available_languages else 0

    voice_lang_display = st.selectbox(
        "Choose Alert Language", 
        options=display_options, 
        index=default_lang_index, 
        help="Select the language for voice alerts and recommendations."
    )
    # Map back to the language code for internal use
    voice_lang = [k for k, v in language_display_names.items() if v == voice_lang_display][0]


# --- Load and Display Sensor Data ---
df = fetch_sensor_data()

if df.empty:
    st.warning("No data available from Firebase. Please ensure your sensor sends data or check Firebase connection.", icon="⚠️")
else:
    # Get latest sensor data for gauges and current readings
    latest_data = df.iloc[-1].to_dict()

    st.markdown("---")
    st.subheader("📊 Current Sensor Readings")
    # Display key metrics using gauge charts
    gauge_cols = st.columns(4)

    # Helper to create a gauge chart
    def create_gauge(title, value, max_value, suffix, color='green', threshold=None, font_color='white'):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            title={'text': title, 'font': {'color': font_color}}, # Set title font color
            number={'font': {'color': font_color}}, # Set number font color - MOVED HERE
            gauge={
                'axis': {'range': [None, max_value], 'tickwidth': 1, 'tickcolor': font_color}, # Set tick color
                'bar': {'color': color},
                'bgcolor': "white", # This is the background of the gauge itself, not the number
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, max_value * 0.3], 'color': "lightgray"},
                    {'range': [max_value * 0.3, max_value * 0.7], 'color': "gray"},
                    {'range': [max_value * 0.7, max_value], 'color': "darkgray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': threshold if threshold is not None else value 
                },
            }
        ))
        fig.update_layout(
            height=250, 
            margin=dict(l=10, r=10, t=50, b=10), 
            font={'color': font_color, 'family': "Arial"}, # Overall font color for text elements
            paper_bgcolor="rgba(0,0,0,0)", # Transparent background for the plot area
            plot_bgcolor="rgba(0,0,0,0)" # Transparent background for the plot area
        )
        return fig

    # Determine font color based on Streamlit's theme (assuming dark mode)
    # In Streamlit, it's hard to directly detect theme, so we'll assume dark mode for better contrast.
    gauge_font_color = 'white' 

    # Soil Moisture Gauge
    soil_moisture_val = latest_data.get('soil_moisture')
    if soil_moisture_val is not None and not pd.isna(soil_moisture_val):
        with gauge_cols[0]:
            st.plotly_chart(create_gauge("Soil Moisture (%)", soil_moisture_val, 100, "%", 'rgba(0,128,0,0.8)', font_color=gauge_font_color), use_container_width=True)
    else:
        with gauge_cols[0]: st.info("Soil Moisture N/A", icon="ℹ️")

    # Temperature Gauge
    temp_val = latest_data.get('temperature')
    if temp_val is not None and not pd.isna(temp_val):
        with gauge_cols[1]:
            st.plotly_chart(create_gauge("Temperature (°C)", temp_val, 40, "°C", 'rgba(255,165,0,0.8)', font_color=gauge_font_color), use_container_width=True)
    else:
        with gauge_cols[1]: st.info("Temperature N/A", icon="ℹ️")

    # pH Gauge
    ph_val = latest_data.get('ph') # Use 'ph' after processing
    if ph_val is not None and not pd.isna(ph_val):
        with gauge_cols[2]:
            st.plotly_chart(create_gauge("pH", ph_val, 14, "", 'rgba(0,0,255,0.8)', font_color=gauge_font_color), use_container_width=True)
    else:
        with gauge_cols[2]: st.info("pH N/A", icon="ℹ️")

    # Humidity Gauge
    humidity_val = latest_data.get('humidity')
    if humidity_val is not None and not pd.isna(humidity_val):
        with gauge_cols[3]:
            st.plotly_chart(create_gauge("Humidity (%)", humidity_val, 100, "%", 'rgba(128,0,128,0.8)', font_color=gauge_font_color), use_container_width=True)
    else:
        with gauge_cols[3]: st.info("Humidity N/A", icon="ℹ️")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Sensor Trends Over Time")
        # Ensure 'ph' is used for plotting if it was consistently mapped from 'pH'
        plot_features = ['soil_moisture', 'temperature', 'humidity', 'ph', 'light_intensity', 'N', 'P', 'K', 'rainfall', 'growth_factor'] # Removed 'growth_trigger' as per trainer
        existing_plot_features = [f for f in plot_features if f in df.columns]
        
        plot_df_melted = df.dropna(subset=existing_plot_features + ['timestamp']).melt(
            id_vars=['timestamp'], 
            value_vars=existing_plot_features,
            var_name='Sensor Metric',
            value_name='Reading'
        )

        if not plot_df_melted.empty and len(existing_plot_features) > 0:
            try:
                fig = px.line(
                    plot_df_melted,
                    x='timestamp',
                    y='Reading',
                    color='Sensor Metric', 
                    labels={'Reading': 'Sensor Reading', 'timestamp': 'Time'},
                    title="Historical Sensor Readings",
                    template="plotly_dark" # Professional theme
                )
                fig.update_layout(
                    hovermode="x unified",
                    xaxis_title="Time",
                    yaxis_title="Sensor Reading",
                    legend_title="Metric",
                    font=dict(family="Inter", size=12, color="#ffffff"), # Consistent font
                    margin=dict(l=40, r=40, t=60, b=40) # Adjust margins
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error plotting sensor trends: {e}", icon="❌")
                st.warning("⚠️ Could not plot all sensor trends. Check data types or missing values, or if the data is too sparse.", icon="⚠️")
        else:
            st.warning("⚠️ Not enough complete data available for plotting sensor trends. Check if sensors are reporting data for these features.", icon="⚠️")

    with col2:
        st.subheader("🌿 Crop Care Recommendations")
        selected_crop_type = st.selectbox("Select Growing Crop", all_crop_labels if all_crop_labels else ["No crops found"], key="crop_select")
        
        if df is not None and not df.empty and selected_crop_type:
            care_tips = crop_care_advice(df, selected_crop_type, lang=voice_lang) # Pass selected language
            st.markdown("---")
            for tip in care_tips:
                st.write(tip)
            
            if st.button(f"🔊 Play Top Alerts ({language_display_names.get(voice_lang, voice_lang)})"):
                if care_tips:
                    for i, tip in enumerate(care_tips[:2]): # Play up to 2 alerts
                        # Remove markdown for better speech, and also remove emojis
                        clean_tip = tip.replace('**', '').replace('🌱', '').replace('💧', '').replace('🌡️', '').replace('💨', '').replace('🧪', '').replace('☀️', '').replace('🌧️', '').replace('✅', '').strip()
                        if clean_tip: # Only play if there's actual text after cleaning
                            st.info(f"Playing alert {i+1}: {clean_tip}", icon="🔊")
                            speak_tip(clean_tip, lang=voice_lang)
                else:
                    st.info("No specific alerts to play.", icon="ℹ️")

        elif not selected_crop_type:
            st.info("Please select a crop to get recommendations.", icon="ℹ️")
        else:
            st.info("No sensor data available for crop care recommendations.", icon="ℹ️")

        st.subheader("🤖 AI-Based Growth Prediction")
        soil_moisture_pred, light_intensity_pred, nutrient_sum_pred = None, None, None
        if df is not None and not df.empty and selected_crop_type and model is not None and input_scaler is not None and output_scaler is not None and crop_encoder is not None:
            soil_moisture_pred, light_intensity_pred, nutrient_sum_pred = predict_growth(df, selected_crop_type)
            if soil_moisture_pred is not None:
                if 0 <= soil_moisture_pred <= 100:
                    st.success(f"📊 Predicted Soil Moisture: **{soil_moisture_pred:.2f}%**", icon="📊")
                else:
                    st.warning(f"📊 Predicted Soil Moisture: **{soil_moisture_pred:.2f}%**. This value seems unusual (Expected between 0-100%).", icon="⚠️")
                st.info(f"💡 Predicted Light Intensity: **{light_intensity_pred:.2f} lux**", icon="💡")
                st.info(f"🌿 Predicted NPK Nutrient Sum: **{nutrient_sum_pred:.2f}**", icon="🌿")
            else:
                st.info("Not enough data or issue with model prediction. Check logs above for details.", icon="ℹ️")
        else:
            st.info("Select a crop, ensure sensor data is available, and all AI components are loaded for prediction.", icon="ℹ️")

        st.subheader("📉 Market Price Forecast")
        if df is not None and not df.empty and selected_crop_type and market_price_model is not None and market_crop_encoder is not None:
            latest_sensor_data_for_price = df.iloc[-1].to_dict()
            predicted_price = predict_market_price(latest_sensor_data_for_price, selected_crop_type, market_price_model, market_crop_encoder, market_price_features)
            if predicted_price is not None:
                st.success(f"💰 Estimated Market Price for {selected_crop_type}: **₹ {predicted_price:.2f} / unit**", icon="💰")
            else:
                st.info("Cannot forecast market price. Ensure all required sensor data is available and market model is trained.", icon="ℹ️")
        else:
            st.info("Select a crop, ensure sensor data is available, and market model is trained for market price forecast.", icon="ℹ️")


        st.subheader("🌾 Crop Suggestion Based on Predicted Conditions")
        if soil_moisture_pred is not None and not pd.isna(soil_moisture_pred) and 0 <= soil_moisture_pred <= 100:
            latest_sensor_data_for_suggestion = df.iloc[-1].to_dict()
            # Ensure 'ph' is used from the processed data for the recommender
            current_ph = latest_sensor_data_for_suggestion.get('ph') 
            current_temp = latest_sensor_data_for_suggestion.get('temperature')
            current_rainfall = latest_sensor_data_for_suggestion.get('rainfall')

            if all(v is not None and not pd.isna(v) for v in [current_ph, current_temp, current_rainfall]):
                seed_recommendation = recommend_seeds(current_ph, current_temp, current_rainfall, soil_moisture_pred, lang=voice_lang) # Pass selected language
                st.write(seed_recommendation)
            else:
                st.info("Missing essential current sensor data (pH, temperature, rainfall) for crop suggestions.", icon="ℹ️")
        else:
            st.info("Predicted soil moisture is out of typical range or not available, hindering specific crop suggestions.", icon="ℹ️")

    # --- Real-Time Plant Monitoring (Fetched from Firebase) ---
    st.subheader("🌿 Real-Time Plant Monitoring (Simulated)")
    camera_data = fetch_camera_feed_data()
    if camera_data:
        st.write(f"🕒 Timestamp: {camera_data.get('timestamp', 'N/A')}")
        st.success(f"📈 Growth Stage: {camera_data.get('stage', 'N/A')}", icon="📈")
        st.warning(f"⚠️ Advisory: {camera_data.get('alert', 'N/A')}", icon="⚠️")
    else:
        st.info("No real-time plant monitoring data available from Firebase. Please ensure the dummy camera simulator is running and pushing data.", icon="ℹ️")

    st.markdown("---")
    st.subheader("📋 Latest Sensor Readings (Raw Data)")
    if not df.empty:
        st.dataframe(df.tail(10))
    else:
        st.info("No sensor data to display.", icon="ℹ️")
    
    st.markdown("---")
    # Display initialization status messages at the very bottom
    with st.expander("Application Initialization Status"):
        for msg in firebase_init_status:
            st.write(msg)
