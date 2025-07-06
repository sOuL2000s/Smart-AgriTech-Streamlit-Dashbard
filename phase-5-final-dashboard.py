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
import base64
import tempfile
import os
import json
import joblib # For saving/loading scalers

# For Voice Alerts
from gtts import gTTS

# Check for playsound availability
PLAYSOUND_AVAILABLE = False
try:
    import playsound
    PLAYSOUND_AVAILABLE = True
except ImportError:
    st.warning("`playsound` library not found. Voice alerts will be generated but not played. "
               "Install with `pip install playsound` for local playback. "
               "For cloud deployment, consider embedding HTML audio.")
except Exception as e:
    st.warning(f"Error importing playsound: {e}. Voice alerts might not work.")

# --- Firebase Secure Setup (Render-Compatible) ---
firebase_key_b64 = os.getenv("FIREBASE_KEY_B64")
firebase_cred_path = None # Initialize to None

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
    st.success("✅ Firebase initialized successfully.")
except Exception as e:
    st.error(f"❌ Firebase initialization failed: {e}")
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
    st.success(f"✅ Crop labels loaded: {len(all_crop_labels)} unique crops found.")
except FileNotFoundError:
    st.error("❌ 'cleaned_sensor_data.csv' not found. Please ensure it's in the same directory.")
    all_crop_labels = [] # Initialize as empty to prevent errors later
    # Fallback encoder, might not be fully representative without actual data
    crop_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False) 
except Exception as e:
    st.error(f"❌ Error loading 'cleaned_sensor_data.csv': {e}")
    all_crop_labels = []
    # Fallback encoder
    crop_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False) 

# --- Load AI Model ---
model = None
try:
    model = tf.keras.models.load_model("tdann_pnsm_model.keras")
    st.success("✅ AI model (tdann_pnsm_model.keras) loaded successfully.")
except Exception as e:
    st.error(f"❌ Error loading AI model (tdann_pnsm_model.keras): {e}")
    st.stop() # Stop the app if the model cannot be loaded

# --- Load Scalers ---
# IMPORTANT: These scalers MUST be the ones fitted during the model training phase.
input_scaler = None
output_scaler = None
try:
    input_scaler = joblib.load('tdann_input_scaler.joblib')
    output_scaler = joblib.load('tdann_output_scaler.joblib')
    st.success("✅ Input and Output scalers loaded successfully.")
except FileNotFoundError:
    st.error("❌ Scaler files (tdann_input_scaler.joblib, tdann_output_scaler.joblib) not found. "
             "The model predictions might be inaccurate without the correct scalers. "
             "Please ensure they are saved during model training and placed in the same directory.")
    # In a real production environment, you might want to stop the app here or handle robustly.
    input_scaler = MinMaxScaler() # Fallback: Initialize new scalers, but warn the user.
    output_scaler = MinMaxScaler() # Fallback: Initialize new scalers, but warn the user.
    st.warning("⚠️ Proceeding with newly initialized scalers. Predictions may be inaccurate.")
except Exception as e:
    st.error(f"❌ Error loading scalers: {e}")
    input_scaler = MinMaxScaler() # Fallback
    output_scaler = MinMaxScaler() # Fallback
    st.warning("⚠️ Proceeding with newly initialized scalers. Predictions may be inaccurate.")


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
        st.error("Cannot train market price model: Crop encoder not initialized.")
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
    st.success("✅ Market price prediction model trained (simulated data).")
else:
    st.error("❌ Market price prediction model could not be trained.")


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
                st.warning(f"Skipping non-dict entry in Firebase: {key}: {value}")
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
        st.warning("pH value missing/NaN from Firebase. Imputing with default pH 6.5 for prediction.")
    # --- END NEW ---

    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    df = df.sort_values('timestamp')
    return df.reset_index(drop=True)

# --- Predict Growth (Multi-Output TDANN) ---
def predict_growth(df_latest_data, selected_crop_type):
    """
    Predicts soil moisture, light intensity, and nutrient sum using the loaded AI model.
    Assumes the model was trained with specific input features and multiple outputs.
    """
    if model is None or input_scaler is None or output_scaler is None or crop_encoder is None:
        st.error("AI model or scalers or encoder not loaded. Cannot predict growth.")
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
        st.error(f"Missing expected TDANN input features in sensor data: {missing}. Cannot predict growth.")
        st.info("Ensure Firebase is sending all required sensor data: N, P, K, temperature, humidity, pH/ph, rainfall, growth_factor.")
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
        st.info(f"Not enough complete data points ({len(processed_data_for_prediction)} < {LOOKBACK_WINDOW}) even after filling NaNs. Need at least {LOOKBACK_WINDOW} consecutive entries with non-NaNs initially.")
        st.info("Please ensure enough historical sensor data is available in Firebase for the lookback window.")
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
    st.write(f"DEBUG: Scaler expects {input_scaler.n_features_in_} features.")
    st.write(f"DEBUG: Current input sequence shape: {full_input_features_sequence_np.shape}")
    st.write(f"DEBUG: Expected full input features order: {expected_full_input_features_order}")

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
        st.error(f"Error during AI prediction: {e}")
        st.exception(e) # Display full traceback for debugging
        return None, None, None

# --- Predict Market Price ---
def predict_market_price(latest_data, selected_crop_type, market_model, market_crop_encoder, market_price_features):
    """
    Predicts the market price based on latest sensor data and crop type.
    """
    if market_model is None or market_crop_encoder is None:
        st.error("Market prediction model or encoder not initialized.")
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
            st.warning(f"Missing or NaN feature '{feature}' for market price prediction. Imputing with 0.")
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
        st.error(f"Error during market price prediction: {e}")
        st.exception(e) # Display full traceback for debugging
        return None


# --- Crop Care Advice Function ---
def crop_care_advice(df, crop_type):
    """Provides crop-specific care advice based on latest sensor readings."""
    if df.empty:
        return ["No sensor data available to provide advice."]
    
    latest = df.iloc[-1].to_dict()
    tips = []
    
    ct = crop_type.lower()

    npk_advice = {
        'N': {'min': 50, 'max': 150, 'low_msg': "Consider applying nitrogen-rich fertilizer.", 'high_msg': "Excess nitrogen can promote leafy growth over fruit/flower development."},
        'P': {'min': 20, 'max': 60, 'low_msg': "Consider applying phosphorus fertilizer for root development.", 'high_msg': "High phosphorus can lock up other nutrients."},
        'K': {'min': 50, 'max': 200, 'low_msg': "Consider applying potassium fertilizer for overall plant health and fruit quality.", 'high_msg': "Excess potassium can interfere with calcium and magnesium uptake."},
    }

    for nutrient, thresholds in npk_advice.items():
        if nutrient in latest and not pd.isna(latest.get(nutrient)):
            value = latest[nutrient]
            if value < thresholds['min']:
                tips.append(f"🌱 **{nutrient} is low ({value:.1f})**: {thresholds['low_msg']}")
            elif value > thresholds['max']:
                tips.append(f"🌱 **{nutrient} is high ({value:.1f})**: {thresholds['high_msg']}")

    # Specific crop advice
    if 'soil_moisture' in latest and not pd.isna(latest.get('soil_moisture')):
        sm = latest['soil_moisture']
        if ct == 'wheat':
            if sm < 35: tips.append(f"💧 **Soil Moisture is low ({sm:.1f}%)**: Irrigate lightly – wheat needs 35–50% soil moisture.")
        elif ct == 'rice':
            if sm < 60: tips.append(f"💧 **Soil Moisture is low ({sm:.1f}%)**: Rice needs high moisture. Ensure proper irrigation.")
        elif ct == 'maize':
            if sm < 40: tips.append(f"💧 **Soil Moisture is low ({sm:.1f}%)**: Maize needs moderate soil moisture levels.")
        elif ct == 'banana':
            if sm < 50: tips.append(f"💧 **Soil Moisture is low ({sm:.1f}%)**: Keep soil consistently moist for banana.")
        elif ct == 'mango':
            if sm > 60: tips.append(f"💧 **Soil Moisture is high ({sm:.1f}%)**: Avoid waterlogging. Mango needs well-drained soil.")
        elif ct == 'grapes':
            if sm > 50: tips.append(f"💧 **Soil Moisture is high ({sm:.1f}%)**: Grapes prefer drier soil – avoid overwatering.")
        elif ct == 'cotton':
            if sm < 30: tips.append(f"💧 **Soil Moisture is low ({sm:.1f}%)**: Cotton requires moderate moisture during flowering.")
        elif ct == 'millet' or ct == 'sorghum':
            if sm < 25: tips.append(f"💧 **Soil Moisture is low ({sm:.1f}%)**: These are drought-resistant crops but still need minimal moisture.")
        elif ct == 'jute':
            if sm < 50: tips.append(f"💧 **Soil Moisture is low ({sm:.1f}%)**: Jute requires ample moisture during growth.")
        elif ct == 'pomegranate':
            if sm > 50: tips.append(f"💧 **Soil Moisture is high ({sm:.1f}%)**: Avoid overwatering pomegranate.")
        elif ct == 'muskmelon' or ct == 'watermelon':
            if sm < 30: tips.append(f"💧 **Soil Moisture is low ({sm:.1f}%)**: Melons need consistent watering, especially during fruiting.")
        elif ct == 'coconut':
            if sm < 50: tips.append(f"💧 **Soil Moisture is low ({sm:.1f}%)**: Coconut palms need high moisture levels.")
        elif ct == 'mothbeans':
            if sm < 25: tips.append(f"💧 **Soil Moisture is low ({sm:.1f}%)**: Mothbeans are drought-tolerant but need minimal irrigation during flowering.")
        elif ct == 'mungbean':
            if sm < 30: tips.append(f"💧 **Soil Moisture is low ({sm:.1f}%)**: Ensure regular irrigation during flowering and pod formation.")
        elif ct == 'blackgram':
            if sm < 35: tips.append(f"💧 **Soil Moisture is low ({sm:.1f}%)**: Maintain moderate moisture especially during flowering.")
        elif ct == 'lentil':
            if sm < 25: tips.append(f"💧 **Soil Moisture is low ({sm:.1f}%)**: Lentils need low to moderate moisture.")
        # General advice if not crop-specific
        if sm < 30: tips.append(f"💧 **Soil Moisture is low ({sm:.1f}%)**: General advice: Consider irrigation to prevent drought stress.")
        elif sm > 70: tips.append(f"💧 **Soil Moisture is high ({sm:.1f}%)**: General advice: Ensure good drainage to prevent waterlogging.")

    if 'temperature' in latest and not pd.isna(latest.get('temperature')):
        temp = latest['temperature']
        if ct == 'wheat':
            if temp > 32: tips.append(f"🌡️ **Temperature is high ({temp:.1f}°C)**: Provide shade or irrigate in evening – temp is too high for wheat.")
        elif ct == 'rice':
            if temp > 38: tips.append(f"🌡️ **Temperature is high ({temp:.1f}°C)**: Too hot for rice. Consider evening irrigation or shade.")
        elif ct == 'maize':
            if temp < 20: tips.append(f"🌡️ **Temperature is low ({temp:.1f}°C)**: Maize prefers warm weather (20–30°C).")
        elif ct == 'banana':
            if temp < 15: tips.append(f"🌡️ **Temperature is low ({temp:.1f}°C)**: Banana is sensitive to cold – ensure warm conditions.")
        elif ct == 'mango':
            if temp < 20: tips.append(f"🌡️ **Temperature is low ({temp:.1f}°C)**: Mango requires warmer temperatures (>20°C).")
        elif ct == 'cotton':
            if temp < 20: tips.append(f"🌡️ **Temperature is low ({temp:.1f}°C)**: Cotton thrives in warm temperatures.")
        elif ct == 'millet' or ct == 'sorghum':
            if temp < 20: tips.append(f"🌡️ **Temperature is low ({temp:.1f}°C)**: Warm climate is ideal for millet/sorghum.")
        elif ct == 'coffee':
            if temp < 18: tips.append(f"🌡️ **Temperature is low ({temp:.1f}°C)**: Coffee thrives in 18–24°C range.")
        elif ct == 'jute':
            if temp < 25: tips.append(f"🌡️ **Temperature is low ({temp:.1f}°C)**: Jute grows well in 25–30°C.")
        elif ct == 'papaya':
            if temp < 20: tips.append(f"🌡️ **Temperature is low ({temp:.1f}°C)**: Papaya prefers 21–33°C range.")
        elif ct == 'pomegranate':
            if temp < 20: tips.append(f"🌡️ **Temperature is low ({temp:.1f}°C)**: Ideal temperature is above 20°C.")
        elif ct == 'muskmelon' or ct == 'watermelon':
            if temp < 25: tips.append(f"🌡️ **Temperature is low ({temp:.1f}°C)**: Ensure temperature is warm (>25°C).")
        elif ct == 'coconut':
            if temp < 25: tips.append(f"🌡️ **Temperature is low ({temp:.1f}°C)**: Ideal temperature for coconut is above 25°C.")
        elif ct == 'mothbeans':
            if temp < 22: tips.append(f"🌡️ **Temperature is low ({temp:.1f}°C)**: Temperature should be above 22°C.")
        elif ct == 'mungbean':
            if temp < 20: tips.append(f"🌡️ **Temperature is low ({temp:.1f}°C)**: Mungbean requires warm conditions for optimal growth.")
        elif ct == 'blackgram':
            if temp < 18: tips.append(f"🌡️ **Temperature is low ({temp:.1f}°C)**: Ideal temperature range is 25–35°C.")
        elif ct == 'lentil':
            if temp < 15: tips.append(f"🌡️ **Temperature is low ({temp:.1f}°C)**: Lentils grow well in 18–30°C.")
        # General advice
        if temp < 18: tips.append(f"🌡️ **Temperature is low ({temp:.1f}°C)**: General advice: Cold temperatures can stunt growth. Consider protective measures.")
        elif temp > 35: tips.append(f"🌡️ **Temperature is high ({temp:.1f}°C)**: General advice: High temperatures can cause heat stress. Ensure adequate water and shade.")

    if 'humidity' in latest and not pd.isna(latest.get('humidity')):
        hum = latest['humidity']
        if ct == 'wheat':
            if hum > 70: tips.append(f"💨 **Humidity is high ({hum:.1f}%)**: Watch out for fungal infections – ensure airflow.")
        elif ct == 'rice':
            if hum < 60: tips.append(f"💨 **Humidity is low ({hum:.1f}%)**: Increase ambient humidity or use mulch.")
        elif ct == 'banana':
            if hum < 60: tips.append(f"💨 **Humidity is low ({hum:.1f}%)**: Banana requires high humidity. Consider misting or mulching.")
        elif ct == 'grapes':
            if hum > 70: tips.append(f"💨 **Humidity is high ({hum:.1f}%)**: High humidity may lead to fungal infections.")
        elif ct == 'coffee':
            if hum < 60: tips.append(f"💨 **Humidity is low ({hum:.1f}%)**: Coffee prefers high humidity.")
        elif ct == 'orange':
            if hum > 70: tips.append(f"💨 **Humidity is high ({hum:.1f}%)**: Prune trees to improve airflow and prevent fungal issues.")
        # General advice
        if hum < 40: tips.append(f"💨 **Humidity is low ({hum:.1f}%)**: General advice: Low humidity can cause wilting. Consider misting or increasing soil moisture.")
        elif hum > 80: tips.append(f"💨 **Humidity is high ({hum:.1f}%)**: General advice: High humidity increases risk of fungal diseases. Ensure good ventilation.")

    # Note: Using 'ph' from fetched data after consistency handling
    if 'ph' in latest and not pd.isna(latest.get('ph')):
        ph_val = latest['ph']
        if ct == 'wheat':
            if ph_val < 6.0: tips.append(f"🧪 **pH is low ({ph_val:.1f})**: Slightly acidic – consider applying lime to raise pH.")
        elif ct == 'rice':
            if ph_val < 5.5 or ph_val > 6.5: tips.append(f"🧪 **pH is off ({ph_val:.1f})**: Maintain slightly acidic soil for rice (pH 5.5–6.5).")
        elif ct == 'maize':
            if ph_val < 5.8 or ph_val > 7: tips.append(f"🧪 **pH is off ({ph_val:.1f})**: Maintain soil pH between 5.8–7.0.")
        elif ct == 'papaya':
            if ph_val < 6: tips.append(f"🧪 **pH is low ({ph_val:.1f})**: Slightly acidic to neutral soil is best for papaya.")
        elif ct == 'orange':
            if ph_val < 6 or ph_val > 7.5: tips.append(f"🧪 **pH is off ({ph_val:.1f})**: Ideal soil pH for orange is 6.0–7.5.")
        # General advice
        if ph_val < 5.5: tips.append(f"🧪 **pH is very low ({ph_val:.1f})**: General advice: Soil is too acidic. Apply lime to increase pH and improve nutrient availability.")
        elif ph_val > 7.5: tips.append(f"🧪 **pH is very high ({ph_val:.1f})**: General advice: Soil is too alkaline. Apply sulfur or organic matter to decrease pH.")
        elif not (5.5 <= ph_val <= 7.5): tips.append(f"🧪 **pH is off ({ph_val:.1f})**: General advice: Optimal pH range for most crops is 5.5-7.5. Adjust as needed.")

    if 'light_intensity' in latest and not pd.isna(latest.get('light_intensity')):
        light = latest['light_intensity']
        if ct == 'wheat':
            if light < 400: tips.append(f"☀️ **Light Intensity is low ({light:.1f} lux)**: Ensure the crop gets enough sunlight.")
        elif ct == 'rice':
            if light < 500: tips.append(f"☀️ **Light Intensity is low ({light:.1f} lux)**: Ensure rice gets full sun exposure.")
        # General advice
        if light < 300: tips.append(f"☀️ **Light Intensity is very low ({light:.1f} lux)**: General advice: Insufficient light can hinder photosynthesis. Consider supplemental lighting or pruning.")
        elif light > 800: tips.append(f"☀️ **Light Intensity is very high ({light:.1f} lux)**: General advice: Excessive light can cause scorching. Consider shading during peak hours.")
            
    if 'rainfall' in latest and not pd.isna(latest.get('rainfall')):
        rain = latest['rainfall']
        if rain < 50: 
            tips.append(f"🌧️ **Rainfall is low ({rain:.1f} mm)**: Consider supplementary irrigation, especially for water-intensive crops.")
        elif rain > 200: 
            tips.append(f"🌧️ **Rainfall is high ({rain:.1f} mm)**: Ensure good drainage to prevent waterlogging and root rot.")
        
    return tips if tips else ["✅ All major parameters look good! Keep monitoring regularly for optimal growth."]

# --- Voice Alert Function (Updated for Streamlit Cloud + Local) ---
def speak_tip(tip_text, lang='en'):
    try:
        tts = gTTS(text=tip_text, lang=lang)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            file_path = f.name
            tts.save(file_path)
        
        if PLAYSOUND_AVAILABLE:
            try:
                playsound.playsound(file_path)
            except Exception as e:
                st.error(f"Error playing voice alert with playsound: {e}. Attempting in-browser playback.")
                # Fallback to in-browser playback if playsound fails
                audio_file = open(file_path, "rb")
                audio_bytes = audio_file.read()
                st.audio(audio_bytes, format="audio/mp3", autoplay=True)
        else:
            audio_file = open(file_path, "rb")
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format="audio/mp3", autoplay=True)
    except Exception as e:
        st.error(f"Error generating or playing voice alert: {e}")
        st.info("This might be due to missing audio backend (e.g., `ffplay` on Linux) or `playsound` limitations on web servers.")
    finally:
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path) # Clean up the temporary file

# --- Seed Recommender Function ---
def recommend_seeds(ph, temperature, rainfall, soil_moisture=None):
    """
    Suggests suitable crops based on environmental conditions.
    Args:
        ph (float): Current pH value of the soil.
        temperature (float): Current temperature in Celsius.
        rainfall (float): Recent rainfall in mm.
        soil_moisture (float, optional): Current soil moisture percentage.
                                        If available, provides more specific advice.
    Returns:
        str: Recommended crops or general advice.
    """
    recommendations = []

    # pH based recommendations
    if ph < 5.5:
        recommendations.append("acid-tolerant crops like blueberries, potatoes, or specific rice varieties")
    elif ph > 7.5:
        recommendations.append("alkaline-tolerant crops such as asparagus, spinach, or specific varieties of alfalfa")
    else:
        recommendations.append("a wide range of crops thrive in neutral to slightly acidic pH (5.5-7.5), including wheat, maize, and most vegetables")

    # Temperature based recommendations
    if temperature > 35:
        recommendations.append("heat-tolerant crops like millet, sorghum, cotton, or some varieties of beans")
    elif temperature < 15:
        recommendations.append("cold-hardy crops such as wheat (winter varieties), barley, oats, or peas")
    else:
        recommendations.append("warm-season crops like maize, rice (tropical), most vegetables, and fruits")

    # Rainfall based recommendations
    if rainfall < 50: # Low rainfall
        recommendations.append("drought-resistant crops like millet, sorghum, chickpeas, or certain types of beans (e.g., mothbeans)")
    elif rainfall > 200: # High rainfall, potentially waterlogging
        recommendations.append("water-loving crops such as rice, sugarcane, jute, or crops that tolerate temporary waterlogging")
    else:
        recommendations.append("crops suitable for moderate rainfall, including wheat, maize, and many vegetables")

    # Soil Moisture based recommendations (more granular if available)
    if soil_moisture is not None:
        if soil_moisture < 30: # Very dry
            recommendations.append("very drought-tolerant crops (e.g., desert-adapted melons or some herbs)")
        elif soil_moisture > 80: # Very wet, prone to waterlogging
            recommendations.append("semi-aquatic crops or those highly tolerant to waterlogging (e.g., taro, some rice varieties if poorly drained)")

    if not recommendations:
        return "No specific recommendations, as current conditions are unusual or general."
    
    return "Based on current conditions, you might consider: " + ", ".join(recommendations) + ". Please consult local agricultural experts for precise recommendations."


# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("🌿 Smart AgriTech Dashboard")

# Refresh button
if st.button("🔄 Refresh Data"):
    st.cache_data.clear() # Clear cache to fetch fresh data
    st.rerun()

# Language selector for voice alerts
voice_lang = st.selectbox("Choose Alert Language", ["en", "hi"], index=0, help="English (en) or Hindi (hi)")

# --- Load and Display Sensor Data ---
df = fetch_sensor_data()

if df.empty:
    st.warning("No data available from Firebase. Please ensure your sensor sends data or check Firebase connection.")
else:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Sensor Trends")
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
                    title="Sensor Readings Over Time"
                )
                fig.update_layout(hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error plotting sensor trends: {e}")
                st.warning("⚠️ Could not plot all sensor trends. Check data types or missing values, or if the data is too sparse.")
        else:
            st.warning("⚠️ Not enough complete data available for plotting sensor trends. Check if sensors are reporting data for these features.")

    with col2:
        st.subheader("🌿 Crop Care Recommendations")
        selected_crop_type = st.selectbox("Select Growing Crop", all_crop_labels if all_crop_labels else ["No crops found"], key="crop_select")
        
        if df is not None and not df.empty and selected_crop_type:
            # Display latest sensor data summary
            latest_data_summary = df.iloc[-1].to_dict()
            # Ensure 'ph' is displayed if it's the processed column
            display_data_summary = {k: f"{v:.1f}" if isinstance(v, (int, float)) else v for k, v in latest_data_summary.items() if k not in ['timestamp', 'index', 'pH']}
            st.markdown(f"📊 **Latest Data**: {', '.join([f'{k}: {v}' for k, v in display_data_summary.items()])}")

            care_tips = crop_care_advice(df, selected_crop_type)
            st.markdown("---")
            for tip in care_tips:
                st.write(tip)
            
            if st.button(f"🔊 Play Top Alerts ({'Hindi' if voice_lang=='hi' else 'English'})"):
                if care_tips:
                    for i, tip in enumerate(care_tips[:2]): # Play up to 2 alerts
                        st.info(f"Playing alert {i+1}: {tip}")
                        speak_tip(tip.replace('**', ''), lang=voice_lang) # Remove markdown for better speech
                else:
                    st.info("No specific alerts to play.")

        elif not selected_crop_type:
            st.info("Please select a crop to get recommendations.")
        else:
            st.info("No sensor data available for crop care recommendations.")

        st.subheader("🤖 AI-Based Growth Prediction")
        soil_moisture_pred, light_intensity_pred, nutrient_sum_pred = None, None, None
        if df is not None and not df.empty and selected_crop_type and model is not None and input_scaler is not None and output_scaler is not None and crop_encoder is not None:
            soil_moisture_pred, light_intensity_pred, nutrient_sum_pred = predict_growth(df, selected_crop_type)
            if soil_moisture_pred is not None:
                if 0 <= soil_moisture_pred <= 100:
                    st.success(f"📊 Predicted Soil Moisture: **{soil_moisture_pred:.2f}%**")
                else:
                    st.warning(f"📊 Predicted Soil Moisture: **{soil_moisture_pred:.2f}%**. This value seems unusual (Expected between 0-100%).")
                st.info(f"💡 Predicted Light Intensity: **{light_intensity_pred:.2f} lux**")
                st.info(f"🌿 Predicted NPK Nutrient Sum: **{nutrient_sum_pred:.2f}**")
            else:
                st.info("Not enough data or issue with model prediction. Check logs above for details.")
        else:
            st.info("Select a crop, ensure sensor data is available, and all AI components are loaded for prediction.")

        st.subheader("📉 Market Price Forecast")
        if df is not None and not df.empty and selected_crop_type and market_price_model is not None and market_crop_encoder is not None:
            latest_sensor_data_for_price = df.iloc[-1].to_dict()
            predicted_price = predict_market_price(latest_sensor_data_for_price, selected_crop_type, market_price_model, market_crop_encoder, market_price_features)
            if predicted_price is not None:
                st.success(f"💰 Estimated Market Price for {selected_crop_type}: **₹ {predicted_price:.2f} / unit**")
            else:
                st.info("Cannot forecast market price. Ensure all required sensor data is available and market model is trained.")
        else:
            st.info("Select a crop, ensure sensor data is available, and market model is trained for market price forecast.")


        st.subheader("🌾 Crop Suggestion Based on Predicted Conditions")
        if soil_moisture_pred is not None and not pd.isna(soil_moisture_pred) and 0 <= soil_moisture_pred <= 100:
            latest_sensor_data_for_suggestion = df.iloc[-1].to_dict()
            # Ensure 'ph' is used from the processed data for the recommender
            current_ph = latest_sensor_data_for_suggestion.get('ph') 
            current_temp = latest_sensor_data_for_suggestion.get('temperature')
            current_rainfall = latest_sensor_data_for_suggestion.get('rainfall')

            if all(v is not None and not pd.isna(v) for v in [current_ph, current_temp, current_rainfall]):
                seed_recommendation = recommend_seeds(current_ph, current_temp, current_rainfall, soil_moisture_pred)
                st.write(seed_recommendation)
            else:
                st.info("Missing essential current sensor data (pH, temperature, rainfall) for crop suggestions.")
        else:
            st.info("Predicted soil moisture is out of typical range or not available, hindering specific crop suggestions.")

    st.markdown("---")
    st.subheader("📋 Latest Sensor Readings (Raw Data)")
    if not df.empty:
        st.dataframe(df.tail(10))
    else:
        st.info("No sensor data to display.")