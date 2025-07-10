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

# Check for playsound availability
PLAYSOUND_AVAILABLE = False
try:
    import playsound
    PLAYSOUND_AVAILABLE = True
except ImportError:
    # This warning will be collected and displayed at the bottom
    pass
except Exception as e:
    # This warning will be collected and displayed at the bottom
    pass

# List to collect initialization messages
initialization_messages = []

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
    initialization_messages.append({"type": "success", "message": "Firebase initialized successfully."})
except Exception as e:
    initialization_messages.append({"type": "error", "message": f"Firebase initialization failed: {e}"})
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
    initialization_messages.append({"type": "success", "message": f"Crop labels loaded: {len(all_crop_labels)} unique crops found."})
except FileNotFoundError:
    initialization_messages.append({"type": "error", "message": "'cleaned_sensor_data.csv' not found. Please ensure it's in the same directory."})
    all_crop_labels = [] # Initialize as empty to prevent errors later
    # Fallback encoder, might not be fully representative without actual data
    crop_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False) 
except Exception as e:
    initialization_messages.append({"type": "error", "message": f"Error loading 'cleaned_sensor_data.csv': {e}"})
    all_crop_labels = []
    # Fallback encoder
    crop_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False) 

# --- Load AI Model ---
model = None
try:
    model = tf.keras.models.load_model("tdann_pnsm_model.keras")
    initialization_messages.append({"type": "success", "message": "AI model (tdann_pnsm_model.keras) loaded successfully."})
except Exception as e:
    initialization_messages.append({"type": "error", "message": f"Error loading AI model (tdann_pnsm_model.keras): {e}"})
    st.stop() # Stop the app if the model cannot be loaded

# --- Load Scalers ---
# IMPORTANT: These scalers MUST be the ones fitted during the model training phase.
input_scaler = None
output_scaler = None
try:
    input_scaler = joblib.load('tdann_input_scaler.joblib')
    output_scaler = joblib.load('tdann_output_scaler.joblib')
    initialization_messages.append({"type": "success", "message": "Input and Output scalers loaded successfully."})
except FileNotFoundError:
    initialization_messages.append({"type": "error", "message": "Scaler files (tdann_input_scaler.joblib, tdann_output_scaler.joblib) not found. The model predictions might be inaccurate without the correct scalers. Please ensure they are saved during model training and placed in the same directory."})
    # In a real production environment, you might want to stop the app here or handle robustly.
    input_scaler = MinMaxScaler() # Fallback: Initialize new scalers, but warn the user.
    output_scaler = MinMaxScaler() # Fallback: Initialize new scalers, but warn the user.
    initialization_messages.append({"type": "warning", "message": "Proceeding with newly initialized scalers. Predictions may be inaccurate."})
except Exception as e:
    initialization_messages.append({"type": "error", "message": f"Error loading scalers: {e}"})
    input_scaler = MinMaxScaler() # Fallback
    output_scaler = MinMaxScaler() # Fallback
    initialization_messages.append({"type": "warning", "message": "Proceeding with newly initialized scalers. Predictions may be inaccurate."})


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
        initialization_messages.append({"type": "error", "message": "Cannot train market price model: Crop encoder not initialized."})
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
    initialization_messages.append({"type": "success", "message": "Market price prediction model trained (simulated data)."})
else:
    initialization_messages.append({"type": "error", "message": "Market price prediction model could not be trained."})


# --- Fetch Live Sensor Data ---
@st.cache_data(ttl=10) # Cache data for 10 seconds to reduce Firebase reads
def fetch_sensor_data():
    """Fetches sensor data from Firebase Realtime Database, fetching only the latest record."""
    ref = db.reference('sensors/farm1')
    
    # Fetch only the last record to avoid large data transfers and connection issues
    try:
        snapshot = ref.order_by_key().limit_to_last(1).get()
    except Exception as e:
        initialization_messages.append({"type": "error", "message": f"Error fetching sensor data from Firebase: {e}. This might be due to network issues or very large data. Attempting to fetch historical data for plotting if available."})
        # If fetching latest fails, try to fetch a limited history for plotting
        try:
            snapshot = ref.order_by_key().limit_to_last(50).get() # Fetch last 50 for trends
        except Exception as e_hist:
            initialization_messages.append({"type": "error", "message": f"Failed to fetch historical data: {e_hist}. No sensor data will be displayed."})
            return pd.DataFrame() # Return empty if even historical fetch fails

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
                initialization_messages.append({"type": "warning", "message": f"Skipping non-dict entry in Firebase: {key}: {value}"})
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
        # Changed st.toast to append to initialization_messages
        initialization_messages.append({"type": "warning", "message": "pH value missing/NaN from Firebase. Imputing with default pH 6.5 for prediction."})
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
        st.error(f"Missing expected TDANN input features in sensor data: {set(final_tdann_input_features) - set(available_tdann_features)}. Cannot predict growth.")
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
        'rainfall_low': "🌧️ **Rainfall is low ({rain:.1f} mm)**: {message}",
        'rainfall_high': "🌧️ **Rainfall is high ({rain:.1f} mm)**: {message}",
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
        'rainfall_low_msg': "Consider supplementary irrigation, especially for water-intensive crops.",
        'rainfall_high_msg': "Ensure good drainage to prevent waterlogging and root rot."
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
        'no_data': "No hay datos del sensor disponibles para proporcionar consejos.",
        'npk_low': "🌱 **{nutrient} es bajo ({value:.1f})**: {message}",
        'npk_high': "🌱 **{nutrient} es alto ({value:.1f})**: {message}",
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
        'npk_p_low': "Considere aplicar fertilizante de fósforo para el desarrollo de las raíces.",
        'npk_p_high': "El fósforo alto puede bloquear otros nutrientes.",
        'npk_k_low': "Considere aplicar fertilizante de potasio para la salud general de la planta y la calidad de la fruta.",
        'npk_k_high': "El exceso de potasio puede interferir con la absorción de calcio y magnesio.",
        'wheat_sm_low': "Riegue ligeramente – el trigo necesita 35–50% de humedad del suelo.",
        'rice_sm_low': "El arroz necesita mucha humedad. Asegure un riego adecuado.",
        'maize_sm_low': "El maíz necesita niveles moderados de humedad del suelo.",
        'banana_sm_low': "Mantenga el suelo constantemente húmedo para el plátano.",
        'mango_sm_high': "Evite el encharcamiento. El mango necesita suelo bien drenado.",
        'grapes_sm_high': "Las uvas prefieren suelo más seco – evite el riego excesivo.",
        'cotton_sm_low': "El algodón requiere humedad moderada durante la floración.",
        'millet_sorghum_sm_low': "Estos son cultivos resistentes a la sequía pero aún necesitan humedad mínima.",
        'jute_sm_low': "El yute requiere mucha humedad durante el crecimiento.",
        'pomegranate_sm_high': "Evite regar en exceso la granada.",
        'melon_sm_low': "Los melones necesitan riego constante, especialmente durante la fructificación.",
        'coconut_sm_low': "Las palmas de coco necesitan altos niveles de humedad.",
        'mothbeans_sm_low': "Los frijoles polilla son tolerantes a la sequía pero necesitan riego mínimo durante la floración.",
        'mungbean_sm_low': "Asegure un riego regular durante la floración y la formación de vainas.",
        'blackgram_sm_low': "Mantenga una humedad moderada, especialmente durante la floración.",
        'lentil_sm_low': "Las lentejas necesitan humedad baja a moderada.",
        'general_sm_low': "Consejo general: Considere el riego para prevenir el estrés por sequía.",
        'general_sm_high': "Consejo general: Asegure un buen drenaje para prevenir el encharcamiento.",
        'wheat_temp_high': "Proporcione sombra o riegue por la tarde – la temperatura es demasiado alta para el trigo.",
        'rice_temp_high': "Demasiado calor para el arroz. Considere el riego por la tarde o la sombra.",
        'maize_temp_low': "El maíz prefiere el clima cálido (20–30°C).",
        'banana_temp_low': "El plátano es sensible al frío – asegure condiciones cálidas.",
        'mango_temp_low': "El mango requiere temperaturas más cálidas (>20°C).",
        'cotton_temp_low': "El algodón prospera en temperaturas cálidas.",
        'millet_sorghum_temp_low': "El clima cálido es ideal para el mijo/sorgo.",
        'coffee_temp_low': "El café prospera en el rango de 18–24°C.",
        'jute_temp_low': "El yute crece bien entre 25–30°C.",
        'papaya_temp_low': "La papaya prefiere el rango de 21–33°C.",
        'pomegranate_temp_low': "La temperatura ideal es superior a 20°C.",
        'melon_temp_low': "Asegure que la temperatura sea cálida (>25°C).",
        'coconut_temp_low': "La temperatura ideal para el coco es superior a 25°C.",
        'mothbeans_temp_low': "La temperatura debe ser superior a 22°C.",
        'mungbean_temp_low': "El frijol mungo requiere condiciones cálidas para un crecimiento óptimo.",
        'blackgram_temp_low': "El rango de temperatura ideal es de 25–35°C.",
        'lentil_temp_low': "Las lentejas crecen bien entre 18–30°C.",
        'general_temp_low': "Consejo general: Las bajas temperaturas pueden atrofiar el crecimiento. Considere medidas de protección.",
        'general_temp_high': "Consejo general: Las altas temperaturas pueden causar estrés por calor. Asegure agua y sombra adecuadas.",
        'wheat_hum_high': "Cuidado con las infecciones fúngicas – asegure el flujo de aire.",
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
        'general_light_low': "Consejo general: La luz insuficiente puede dificultar la fotosíntesis. Considere iluminación suplementaria o poda.",
        'general_light_high': "Consejo general: La luz excesiva puede causar quemaduras. Considere la sombra durante las horas pico."
    },
    'fr': { # French
        'no_data': "Aucune donnée de capteur disponible pour fournir des conseils.",
        'npk_low': "🌱 **{nutrient} est bas ({value:.1f})**: {message}",
        'npk_high': "🌱 **{nutrient} est élevé ({value:.1f})**: {message}",
        'soil_moisture_low': "💧 **Humidité du sol basse ({sm:.1f}%)**: {message}",
        'soil_moisture_high': "💧 **Humidité du sol élevée ({sm:.1f}%)**: {message}",
        'temp_low': "🌡️ **Température basse ({temp:.1f}°C)**: {message}",
        'temp_high': "🌡️ **Température élevée ({temp:.1f}°C)**: {message}",
        'humidity_low': "💨 **Humidité basse ({hum:.1f}%)**: {message}",
        'humidity_high': "💨 **Humidité élevée ({hum:.1f}%)**: {message}",
        'ph_low': "🧪 **pH bas ({ph_val:.1f})**: {message}",
        'ph_high': "🧪 **pH élevé ({ph_val:.1f})**: {message}",
        'ph_off': "🧪 **pH incorrect ({ph_val:.1f})**: {message}",
        'light_low': "☀️ **Intensité lumineuse basse ({light:.1f} lux)**: {message}",
        'light_high': "☀️ **Intensité lumineuse élevée ({light:.1f} lux)**: {message}",
        'rainfall_low_msg': "🌧️ **Précipitations faibles ({rain:.1f} mm)**: {message}",
        'rainfall_high_msg': "🌧️ **Précipitations élevées ({rain:.1f} mm)**: {message}",
        'all_good': "✅ Tous les paramètres majeurs semblent bons ! Continuez à surveiller régulièrement pour une croissance optimale.",
        'npk_n_low': "Envisagez d'appliquer un engrais riche en azote.",
        'npk_n_high': "L'excès d'azote peut favoriser la croissance des feuilles au détriment du développement des fruits/fleurs.",
        'npk_p_low': "Envisagez d'appliquer un engrais phosphaté pour le développement des racines.",
        'npk_p_high': "Un niveau élevé de phosphore peut bloquer d'autres nutriments.",
        'npk_k_low': "Envisagez d'appliquer un engrais potassique pour la santé générale de la plante et la qualité des fruits.",
        'npk_k_high': "L'excès de potassium peut interférer avec l'absorption du calcium et du magnésium.",
        'wheat_sm_low': "Irriguez légèrement – le blé a besoin de 35 à 50% d'humidité du sol.",
        'rice_sm_low': "Le riz a besoin d'une humidité élevée. Assurez une irrigation adéquate.",
        'maize_sm_low': "Le maïs a besoin de niveaux d'humidité du sol modérés.",
        'banana_sm_low': "Maintenez le sol constamment humide pour la banane.",
        'mango_sm_high': "Évitez l'engorgement. La mangue a besoin d'un sol bien drainé.",
        'grapes_sm_high': "Les raisins préfèrent un sol plus sec – évitez l'arrosage excessif.",
        'cotton_sm_low': "Le coton nécessite une humidité modérée pendant la floraison.",
        'millet_sorghum_sm_low': "Ce sont des cultures résistantes à la sécheresse mais elles ont toujours besoin d'un minimum d'humidité.",
        'jute_sm_low': "Le jute nécessite une humidité suffisante pendant la croissance.",
        'pomegranate_sm_high': "Évitez d'arroser excessivement la grenade.",
        'melon_sm_low': "Les melons ont besoin d'un arrosage constant, surtout pendant la fructification.",
        'coconut_sm_low': "Les cocotiers ont besoin de niveaux d'humidité élevés.",
        'mothbeans_sm_low': "Les haricots papillon sont tolérants à la sécheresse mais nécessitent une irrigation minimale pendant la floraison.",
        'mungbean_sm_low': "Assurez une irrigation régulière pendant la floraison et la formation des gousses.",
        'blackgram_sm_low': "Maintenez une humidité modérée, surtout pendant la floraison.",
        'lentil_sm_low': "Les lentilles ont besoin d'une humidité faible à modérée.",
        'general_sm_low': "Conseil général : Envisagez l'irrigation pour prévenir le stress hydrique.",
        'general_sm_high': "Conseil général : Assurez un bon drainage pour prévenir l'engorgement.",
        'wheat_temp_high': "Fournissez de l'ombre ou irriguez le soir – la température est trop élevée pour le blé.",
        'rice_temp_high': "Trop chaud pour le riz. Envisagez l'irrigation le soir ou l'ombre.",
        'maize_temp_low': "Le maïs préfère un temps chaud (20–30°C).",
        'banana_temp_low': "La banane est sensible au froid – assurez des conditions chaudes.",
        'mango_temp_low': "La mangue nécessite des températures plus chaudes (>20°C).",
        'cotton_temp_low': "Le coton prospère à des températures chaudes.",
        'millet_sorghum_temp_low': "Le climat chaud est idéal pour le millet/sorgho.",
        'coffee_temp_low': "Le café prospère dans la plage de 18 à 24°C.",
        'jute_temp_low': "Le jute pousse bien entre 25 et 30°C.",
        'papaya_temp_low': "La papaye préfère une plage de 21 à 33°C.",
        'pomegranate_temp_low': "La température idéale est supérieure à 20°C.",
        'melon_temp_low': "Assurez-vous que la température est chaude (>25°C).",
        'coconut_temp_low': "La température idéale pour la noix de coco est supérieure à 25°C.",
        'mothbeans_temp_low': "La température doit être supérieure à 22°C.",
        'mungbean_temp_low': "Le haricot mungo nécessite des conditions chaudes pour une croissance optimale.",
        'blackgram_temp_low': "La plage de température idéale est de 25 à 35°C.",
        'lentil_temp_low': "Les lentilles poussent bien entre 18 et 30°C.",
        'general_temp_low': "Conseil général : Les basses températures peuvent retarder la croissance. Envisagez des mesures de protection.",
        'general_temp_high': "Conseil général : Les températures élevées peuvent provoquer un stress thermique. Assurez un apport suffisant en eau et de l'ombre.",
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
        'general_light_high': "Conseil général : Une lumière excessive peut provoquer des brûlures. Envisagez de l'ombre pendant les heures de pointe."
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
        'all_good': "✅ Alle Hauptparameter sehen gut aus! Regelmäßige Überwachung für optimales Wachstum beibehalten.",
        'npk_n_low': "Stickstoffreichen Dünger in Betracht ziehen.",
        'npk_n_high': "Überschüssiger Stickstoff kann Blattwachstum gegenüber Frucht-/Blütenentwicklung fördern.",
        'npk_p_low': "Phosphordünger für die Wurzelentwicklung in Betracht ziehen.",
        'npk_p_high': "Hoher Phosphor kann andere Nährstoffe blockieren.",
        'npk_k_low': "Kaliumdünger für die allgemeine Pflanzengesundheit und Fruchtqualität in Betracht ziehen.",
        'npk_k_high': "Überschüssiges Kalium kann die Aufnahme von Kalzium und Magnesium beeinträchtigen.",
        'wheat_sm_low': "Leicht bewässern – Weizen benötigt 35–50% Bodenfeuchtigkeit.",
        'rice_sm_low': "Reis benötigt hohe Feuchtigkeit. Sorgen Sie für eine ordnungsgemäße Bewässerung.",
        'maize_sm_low': "Mais benötigt moderate Bodenfeuchtigkeitswerte.",
        'banana_sm_low': "Boden für Bananen konstant feucht halten.",
        'mango_sm_high': "Staunässe vermeiden. Mango benötigt gut durchlässigen Boden.",
        'grapes_sm_high': "Trauben bevorzugen trockeneren Boden – übermäßiges Gießen vermeiden.",
        'cotton_sm_low': "Baumwolle benötigt während der Blüte moderate Feuchtigkeit.",
        'millet_sorghum_sm_low': "Dies sind dürreresistente Pflanzen, benötigen aber dennoch minimale Feuchtigkeit.",
        'jute_sm_low': "Jute benötigt während des Wachstums reichlich Feuchtigkeit.",
        'pomegranate_sm_high': "Granatapfel nicht übermäßig gießen.",
        'melon_sm_low': "Melonen benötigen konstante Bewässerung, besonders während der Fruchtbildung.",
        'coconut_sm_low': "Kokospalmen benötigen hohe Feuchtigkeitswerte.",
        'mothbeans_sm_low': "Mothbohnen sind dürretolerant, benötigen aber während der Blüte minimale Bewässerung.",
        'mungbean_sm_low': "Regelmäßige Bewässerung während der Blüte und Hülsenbildung sicherstellen.",
        'blackgram_sm_low': "Moderate Feuchtigkeit, besonders während der Blüte, aufrechterhalten.",
        'lentil_sm_low': "Linsen benötigen geringe bis moderate Feuchtigkeit.",
        'general_sm_low': "Allgemeiner Ratschlag: Bewässerung in Betracht ziehen, um Dürrestress vorzubeugen.",
        'general_sm_high': "Allgemeiner Ratschlag: Sorgen Sie für eine gute Drainage, um Staunässe zu vermeiden.",
        'wheat_temp_high': "Schatten spenden oder abends bewässern – Temperatur ist zu hoch für Weizen.",
        'rice_temp_high': "Zu heiß für Reis. Abends bewässern oder Schatten spenden.",
        'maize_temp_low': "Mais bevorzugt warmes Wetter (20–30°C).",
        'banana_temp_low': "Banane ist kälteempfindlich – warme Bedingungen sicherstellen.",
        'mango_temp_low': "Mango benötigt wärmere Temperaturen (>20°C).",
        'cotton_temp_low': "Baumwolle gedeiht bei warmen Temperaturen.",
        'millet_sorghum_temp_low': "Warmes Klima ist ideal für Hirse/Sorghum.",
        'coffee_temp_low': "Kaffee gedeiht im Bereich von 18–24°C.",
        'jute_temp_low': "Jute wächst gut bei 25–30°C.",
        'papaya_temp_low': "Papaya bevorzugt den Bereich von 21–33°C.",
        'pomegranate_temp_low': "Ideale Temperatur liegt über 20°C.",
        'melon_temp_low': "Sicherstellen, dass die Temperatur warm ist (>25°C).",
        'coconut_temp_low': "Ideale Temperatur für Kokosnuss liegt über 25°C.",
        'mothbeans_temp_low': "Temperatur sollte über 22°C liegen.",
        'mungbean_temp_low': "Mungbohnen benötigen warme Bedingungen für optimales Wachstum.",
        'blackgram_temp_low': "Idealer Temperaturbereich ist 25–35°C.",
        'lentil_temp_low': "Linsen wachsen gut bei 18–30°C.",
        'general_temp_low': "Allgemeiner Ratschlag: Kalte Temperaturen können das Wachstum hemmen. Schutzmaßnahmen in Betracht ziehen.",
        'general_temp_high': "Allgemeiner Ratschlag: Hohe Temperaturen können Hitzestress verursachen. Ausreichend Wasser und Schatten sicherstellen.",
        'wheat_hum_high': "Auf Pilzinfektionen achten – Luftzirkulation sicherstellen.",
        'rice_hum_low': "Umgebungsfeuchtigkeit erhöhen oder Mulch verwenden.",
        'banana_hum_low': "Banane benötigt hohe Luftfeuchtigkeit. Besprühen oder Mulchen in Betracht ziehen.",
        'grapes_hum_high': "Hohe Luftfeuchtigkeit kann zu Pilzinfektionen führen.",
        'coffee_hum_low': "Kaffee bevorzugt hohe Luftfeuchtigkeit.",
        'orange_hum_high': "Bäume beschneiden, um die Luftzirkulation zu verbessern und Pilzprobleme zu vermeiden.",
        'general_hum_low': "Allgemeiner Ratschlag: Geringe Luftfeuchtigkeit kann Welken verursachen. Besprühen oder Erhöhung der Bodenfeuchtigkeit in Betracht ziehen.",
        'general_hum_high': "Allgemeiner Ratschlag: Hohe Luftfeuchtigkeit erhöht das Risiko von Pilzkrankheiten. Sorgen Sie für gute Belüftung.",
        'wheat_ph_low': "Leicht sauer – Kalk auftragen, um den pH-Wert zu erhöhen.",
        'rice_ph_off': "Leicht sauren Boden für Reis beibehalten (pH 5.5–6.5).",
        'maize_ph_off': "Boden-pH-Wert zwischen 5.8–7.0 halten.",
        'papaya_ph_low': "Leicht saurer bis neutraler Boden ist am besten für Papaya.",
        'orange_ph_off': "Idealer Boden-pH-Wert für Orangen ist 6.0–7.5.",
        'general_ph_very_low': "Allgemeiner Ratschlag: Der Boden ist zu sauer. Kalk auftragen, um den pH-Wert zu erhöhen und die Nährstoffverfügbarkeit zu verbessern.",
        'general_ph_very_high': "Allgemeiner Ratschlag: Der Boden ist zu alkalisch. Schwefel oder organische Substanz auftragen, um den pH-Wert zu senken.",
        'general_ph_off': "Allgemeiner Ratschlag: Der optimale pH-Bereich für die meisten Pflanzen liegt bei 5.5-7.5. Ajuste según sea necesario.",
        'wheat_light_low': "Stellen Sie sicher, dass die Pflanze ausreichend Sonnenlicht erhält.",
        'rice_light_low': "Stellen Sie sicher, dass Reis volle Sonneneinstrahlung erhält.",
        'general_light_low': "Allgemeiner Ratschlag: Unzureichendes Licht kann die Photosynthese behindern. Zusätzliche Beleuchtung oder Beschneidung in Betracht ziehen.",
        'general_light_high': "Allgemeiner Ratschlag: Übermäßiges Licht kann Verbrennungen verursachen. Schatten während der Spitzenzeiten in Betracht ziehen."
    },
    'ar': { # Arabic
        'no_data': "لا توجد بيانات استشعار متاحة لتقديم المشورة.",
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
        'all_good': "✅ جميع المعايير الرئيسية تبدو جيدة! استمر في المراقبة بانتظام لتحقيق النمو الأمثل.",
        'npk_n_low': "فكر في استخدام سماد غني بالنيتروجين.",
        'npk_n_high': "النيتروجين الزائد يمكن أن يعزز نمو الأوراق على حساب نمو الفاكهة/الزهور.",
        'npk_p_low': "فكر في استخدام سماد الفوسفور لتنمية الجذور.",
        'npk_p_high': "الفوسفور العالي يمكن أن يمنع امتصاص العناصر الغذائية الأخرى.",
        'npk_k_low': "فكر في استخدام سماد البوتاسيوم للصحة العامة للنبات وجودة الفاكهة.",
        'npk_k_high': "البوتاسيوم الزائد يمكن أن يتداخل مع امتصاص الكالسيوم والمغنيسيوم.",
        'wheat_sm_low': "الري الخفيف – يحتاج القمح إلى 35-50% رطوبة التربة.",
        'rice_sm_low': "يحتاج الأرز إلى رطوبة عالية. تأكد من الري المناسب.",
        'maize_sm_low': "يحتاج الذرة إلى مستويات معتدلة من رطوبة التربة.",
        'banana_sm_low': "حافظ على رطوبة التربة باستمرار للموز.",
        'mango_sm_high': "تجنب التشبع بالمياه. يحتاج المانجو إلى تربة جيدة التصريف.",
        'grapes_sm_high': "تفضل العنب التربة الجافة – تجنب الإفراط في الري.",
        'cotton_sm_low': "يتطلب القطن رطوبة معتدلة أثناء الإزهار.",
        'millet_sorghum_sm_low': "هذه محاصيل مقاومة للجفاف ولكنها لا تزال بحاجة إلى الحد الأدنى من الرطوبة.",
        'jute_sm_low': "يتطلب الجوت رطوبة وفيرة أثناء النمو.",
        'pomegranate_sm_high': "تجنب الإفراط في ري الرمان.",
        'melon_sm_low': "تحتاج البطيخ إلى ري مستمر، خاصة أثناء الإثمار.",
        'coconut_sm_low': "تحتاج أشجار النخيل إلى مستويات رطوبة عالية.",
        'mothbeans_sm_low': "المحاصيل الفولية مقاومة للجفاف ولكنها تحتاج إلى ري قليل أثناء الإزهار.",
        'mungbean_sm_low': "تأكد من الري المنتظم أثناء الإزهار وتكوين القرون.",
        'blackgram_sm_low': "حافظ على رطوبة معتدلة خاصة أثناء الإزهار.",
        'lentil_sm_low': "تحتاج العدس إلى رطوبة منخفضة إلى معتدلة.",
        'general_sm_low': "نصيحة عامة: فكر في الري لمنع إجهاد الجفاف.",
        'general_sm_high': "نصيحة عامة: تأكد من التصريف الجيد لمنع التشبع بالمياه.",
        'wheat_temp_high': "وفر الظل أو الري في المساء – درجة الحرارة مرتفعة جدًا للقمح.",
        'rice_temp_high': "ساخن جدًا للأرز. فكر في الري المسائي أو الظل.",
        'maize_temp_low': "يفضل الذرة الطقس الدافئ (20-30 درجة مئوية).",
        'banana_temp_low': "الموز حساس للبرد – تأكد من توفر ظروف دافئة.",
        'mango_temp_low': "يتطلب المانجو درجات حرارة أكثر دفئًا (>20 درجة مئوية).",
        'cotton_temp_low': "يزدهر القطن في درجات حرارة دافئة.",
        'millet_sorghum_temp_low': "المناخ الدافئ مثالي للدخن/الذرة الرفيعة.",
        'coffee_temp_low': "يزدهر البن في نطاق 18-24 درجة مئوية.",
        'jute_temp_low': "ينمو الجوت جيدًا في 25-30 درجة مئوية.",
        'papaya_temp_low': "تفضل البابايا نطاق 21-33 درجة مئوية.",
        'pomegranate_temp_low': "درجة الحرارة المثالية أعلى من 20 درجة مئوية.",
        'melon_temp_low': "تأكد من أن درجة الحرارة دافئة (>25 درجة مئوية).",
        'coconut_temp_low': "درجة الحرارة المثالية لجوز الهند أعلى من 25 درجة مئوية.",
        'mothbeans_temp_low': "يجب أن تكون درجة الحرارة أعلى من 22 درجة مئوية.",
        'mungbean_temp_low': "يحتاج المونج إلى ظروف دافئة للنمو الأمثل.",
        'blackgram_temp_low': "نطاق درجة الحرارة المثالي هو 25-35 درجة مئوية.",
        'lentil_temp_low': "تنمو العدس جيدًا في 18-30 درجة مئوية.",
        'general_temp_low': "نصيحة عامة: درجات الحرارة الباردة يمكن أن تعيق النمو. فكر في تدابير وقائية.",
        'general_temp_high': "نصيحة عامة: درجات الحرارة المرتفعة يمكن أن تسبب إجهادًا حراريًا. تأكد من توفر الماء والظل الكافيين.",
        'wheat_hum_high': "احذر من الالتهابات الفطرية – تأكد من تدفق الهواء.",
        'rice_hum_low': "زيادة الرطوبة المحيطة أو استخدام النشارة.",
        'banana_hum_low': "يحتاج الموز إلى رطوبة عالية. فكر في الرش أو التغطية بالنشارة.",
        'grapes_hum_high': "قد تؤدي الرطوبة العالية إلى التهابات فطرية.",
        'coffee_hum_low': "يفضل البن الرطوبة العالية.",
        'orange_hum_high': "تقليم الأشجار لتحسين تدفق الهواء ومنع مشاكل الفطريات.",
        'general_hum_low': "نصيحة عامة: الرطوبة المنخفضة يمكن أن تسبب الذبول. فكر في الرش أو زيادة رطوبة التربة.",
        'general_hum_high': "نصيحة عامة: الرطوبة العالية تزيد من خطر الإصابة بالأمراض الفطرية. تأكد من التهوية الجيدة.",
        'wheat_ph_low': "حمضية قليلاً – فكر في استخدام الجير لرفع الرقم الهيدروجيني.",
        'rice_ph_off': "حافظ على تربة حمضية قليلاً للأرز (pH 5.5–6.5).",
        'maize_ph_off': "حافظ على الرقم الهيدروجيني للتربة بين 5.8–7.0.",
        'papaya_ph_low': "التربة الحمضية قليلاً إلى المحايدة هي الأفضل للبابايا.",
        'orange_ph_off': "الرقم الهيدروجيني المثالي للتربة للبرتقال هو 6.0–7.5.",
        'general_ph_very_low': "نصيحة عامة: التربة حمضية جدًا. استخدم الجير لزيادة الرقم الهيدروجيني وتحسين توافر المغذيات.",
        'general_ph_very_high': "نصيحة عامة: التربة قلوية جدًا. استخدم الكبريت أو المواد العضوية لخفض الرقم الهيدروجيني.",
        'general_ph_off': "نصيحة عامة: نطاق الرقم الهيدروجيني الأمثل لمعظم المحاصيل هو 5.5-7.5. اضبط حسب الحاجة.",
        'wheat_light_low': "تأكد من حصول المحصول على ما يكفي من ضوء الشمس.",
        'rice_light_low': "تأكد من حصول الأرز على التعرض الكامل للشمس.",
        'general_light_low': "نصيحة عامة: الضوء غير الكافي يمكن أن يعيق التمثيل الضوئي. فكر في الإضاءة التكميلية أو التقليم.",
        'general_light_high': "نصيحة عامة: الضوء الزائد يمكن أن يسبب الحروق. فكر في التظليل خلال ساعات الذروة."
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

# --- Voice Alert Function (Updated for Streamlit Cloud + Local) ---
def speak_tip(tip_text, lang='en'):
    try:
        with st.spinner(f"Generating voice alert in {lang.upper()}..."):
            tts = gTTS(text=tip_text, lang=lang)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                file_path = f.name
                tts.save(file_path)
            
            if PLAYSOUND_AVAILABLE:
                try:
                    playsound.playsound(file_path)
                except Exception as e:
                    initialization_messages.append({"type": "error", "message": f"Error playing voice alert with playsound: {e}. Attempting in-browser playback."})
                    # Fallback to in-browser playback if playsound fails
                    audio_file = open(file_path, "rb")
                    audio_bytes = audio_file.read()
                    st.audio(audio_bytes, format="audio/mp3", autoplay=True)
            else:
                audio_file = open(file_path, "rb")
                audio_bytes = audio_file.read()
                st.audio(audio_bytes, format="audio/mp3", autoplay=True)
    except Exception as e:
        initialization_messages.append({"type": "error", "message": f"Error generating or playing voice alert: {e}. This might be due to missing audio backend (e.g., `ffplay` on Linux) or `playsound` limitations on web servers."})
    finally:
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path) # Clean up the temporary file

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
        'acid_tolerant': "cultivos tolerantes al ácido como arándanos, patatas o variedades específicas de arroz",
        'alkaline_tolerant': "cultivos tolerantes a la alcalinidad como espárragos, espinacas o variedades específicas de alfalfa",
        'neutral_ph': "una amplia gama de cultivos prospera en pH neutro a ligeramente ácido (5.5-7.5), incluyendo trigo, maíz y la mayoría de las verduras",
        'heat_tolerant': "cultivos tolerantes al calor como mijo, sorgo, algodón o algunas variedades de frijoles",
        'cold_hardy': "cultivos resistentes al frío como trigo (variedades de invierno), cebada, avena o guisantes",
        'warm_season': "cultivos de temporada cálida como maíz, arroz (tropical), la mayoría de las verduras y frutas",
        'drought_resistant': "cultivos resistentes a la sequía como mijo, sorgo, garbanzos o ciertos tipos de frijoles (por ejemplo, frijol polilla)",
        'water_loving': "cultivos amantes del agua como arroz, caña de azúcar, yute o cultivos que toleran el encharcamiento temporal",
        'moderate_rainfall': "cultivos adecuados para precipitaciones moderadas, incluyendo trigo, maíz y muchas verduras",
        'very_dry': "cultivos muy tolerantes a la sequía (por ejemplo, melones adaptados al desierto o algunas hierbas)",
        'very_wet': "cultivos semiacuáticos o aquellos altamente tolerantes al encharcamiento (por ejemplo, taro, algunas variedades de arroz si el drenaje es deficiente)",
        'no_specific': "No hay recomendaciones específicas, ya que las condiciones actuales son inusuales o generales."
    },
    'fr': { # French
        'intro': "Basé sur les conditions actuelles, vous pourriez envisager : ",
        'outro': ". Veuillez consulter des experts agricoles locaux pour des recommandations précises.",
        'acid_tolerant': "cultures tolérantes à l'acide comme les myrtilles, les pommes de terre ou certaines variétés de riz",
        'alkaline_tolerant': "cultures tolérantes à l'alcalin telles que l'asperge, les épinards ou certaines variétés de luzerne",
        'neutral_ph': "une large gamme de cultures prospère dans un pH neutre à légèrement acide (5.5-7.5), y compris le blé, le maïs et la plupart des légumes",
        'heat_tolerant': "cultures tolérantes à la chaleur comme le millet, le sorgho, le coton ou certaines variétés de haricots",
        'cold_hardy': "cultures résistantes au froid telles que le blé (variétés d'hiver), l'orge, l'avoine ou les pois",
        'warm_season': "cultures de saison chaude comme le maïs, le riz (tropical), la plupart des légumes et des fruits",
        'drought_resistant': "cultures résistantes à la sécheresse comme le millet, le sorgho, les pois chiches ou certains types de haricots (par exemple, le haricot papillon)",
        'water_loving': "cultures aimant l'eau telles que le riz, la canne à sucre, le jute ou les cultures qui tolèrent l'engorgement temporaire",
        'moderate_rainfall': "cultures adaptées aux précipitations modérées, y compris le blé, le maïs et de nombreux légumes",
        'very_dry': "cultures très tolérantes à la sécheresse (par exemple, les melons adaptés au désert ou certaines herbes)",
        'very_wet': "cultures semi-aquatiques ou celles très tolérantes à l'engorgement (par exemple, le taro, certaines variétés de riz si le drainage est mauvais)",
        'no_specific': "Aucune recommandation spécifique, car les conditions actuelles sont inhabituelles ou générales."
    },
    'de': { # German
        'intro': "Basierend auf den aktuellen Bedingungen könnten Sie Folgendes in Betracht ziehen: ",
        'outro': ". Bitte konsultieren Sie lokale Landwirtschaftsexperten für präzise Empfehlungen.",
        'acid_tolerant': "säuretolerante Pflanzen wie Heidelbeeren, Kartoffeln oder spezifische Reissorten",
        'alkaline_tolerant': "alkalitolerante Pflanzen wie Spargel, Spinat oder spezifische Luzernesorten",
        'neutral_ph': "eine breite Palette von Pflanzen gedeiht bei neutralem bis leicht saurem pH-Wert (5,5-7,5), einschließlich Weizen, Mais und den meisten Gemüsesorten",
        'heat_tolerant': "hitzetolerante Pflanzen wie Hirse, Sorghum, Baumwolle oder einige Bohnensorten",
        'cold_hardy': "kälteresistente Pflanzen wie Weizen (Winter сорта), Gerste, Hafer oder Erbsen",
        'warm_season': "Warmwetterpflanzen wie Mais, Reis (tropisch), die meisten Gemüsesorten und Früchte",
        'drought_resistant': "dürreresistente Pflanzen wie Hirse, Sorghum, Kichererbsen oder bestimmte Bohnensorten (z. B. Mottenbohnen)",
        'water_loving': "wasserliebende Pflanzen wie Reis, Zuckerrohr, Jute oder Pflanzen, die vorübergehende Staunässe vertragen",
        'moderate_rainfall': "Pflanzen, die für moderate Niederschläge geeignet sind, einschließlich Weizen, Mais und viele Gemüsesorten",
        'very_dry': "sehr dürretolerante Pflanzen (z. B. wüstenangepasste Melonen oder einige Kräuter)",
        'very_wet': "halbwässrige Pflanzen oder solche, die sehr tolerant gegenüber Staunässe sind (z. B. Taro, einige Reissorten bei schlechter Drainage)",
        'no_specific': "Keine spezifischen Empfehlungen, da die aktuellen Bedingungen ungewöhnlich oder allgemein sind."
    },
    'ar': { # Arabic
        'intro': "بناءً على الظروف الحالية، قد تفكر في: ",
        'outro': ". يرجى استشارة خبراء الزراعة المحليين للحصول على توصيات دقيقة.",
        'acid_tolerant': "محاصيل مقاومة للحموضة مثل التوت الأزرق، البطاطس، أو أصناف أرز محددة",
        'alkaline_tolerant': "محاصيل مقاومة للقلوية مثل الهليون، السبانخ، أو أصناف البرسيم الحجازي المحددة",
        'neutral_ph': "تزدهر مجموعة واسعة من المحاصيل في درجة حموضة محايدة إلى حمضية قليلاً (5.5-7.5)، بما في ذلك القمح، الذرة، ومعظم الخضروات",
        'heat_tolerant': "محاصيل مقاومة للحرارة مثل الدخن، الذرة الرفيعة، القطن، أو بعض أصناف الفول",
        'cold_hardy': "محاصيل مقاومة للبرد مثل القمح (أصناف الشتاء)، الشعير، الشوفان، أو البازلاء",
        'warm_season': "محاصيل الموسم الدافئ مثل الذرة، الأرز (الاستوائي)، معظم الخضروات، والفواكه",
        'drought_resistant': "محاصيل مقاومة للجفاف مثل الدخن، الذرة الرفيعة، الحمص، أو أنواع معينة من الفول (مثل فاصوليا الماث)",
        'water_loving': "محاصيل محبة للماء مثل الأرز، قصب السكر، الجوت، أو المحاصيل التي تتحمل التشبع بالمياه مؤقتًا",
        'moderate_rainfall': "محاصيل مناسبة للأمطار المعتدلة، بما في ذلك القمح، الذرة، والعديد من الخضروات",
        'very_dry': "محاصيل شديدة التحمل للجفاف (مثل البطيخ المتكيف مع الصحراء أو بعض الأعشاب)",
        'very_wet': "محاصيل شبه مائية أو تلك شديدة التحمل للتشبع بالمياه (مثل القلقاس، بعض أصناف الأرز إذا كانت التربة سيئة التصريف)",
        'no_specific': "لا توجد توصيات محددة، حيث أن الظروف الحالية غير عادية أو عامة."
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
        lang (str): Language for recommendations ('en' for English, 'hi' for Hindi, etc.).
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
        st.toast("Data refreshed!") # Give user feedback
with col_lang:
    # Get all available languages from ADVICE_MESSAGES
    available_languages = list(ADVICE_MESSAGES.keys())
    # Map language codes to display names for better UX
    lang_display_names = {
        'en': 'English',
        'hi': 'Hindi',
        'es': 'Español',
        'fr': 'Français',
        'de': 'Deutsch',
        'ar': 'العربية'
    }
    # Create a list of (display_name, code) tuples for the selectbox
    lang_options = [(lang_display_names.get(code, code.upper()), code) for code in available_languages]
    # Find the index for 'en' or 'hi' if they exist, otherwise default to 0
    default_lang_index = next((i for i, (name, code) in enumerate(lang_options) if code == 'en'), 0)
    
    selected_lang_name, voice_lang = st.selectbox(
        "Choose Alert Language", 
        options=lang_options, 
        format_func=lambda x: x[0], # Display the name
        index=default_lang_index, 
        help="Select the language for crop care advice and voice alerts."
    )


# --- Load and Display Sensor Data ---
df = fetch_sensor_data()

if df.empty:
    st.warning("No data available from Firebase. Please ensure your sensor sends data or check Firebase connection.")
else:
    # Get latest sensor data for gauges and current readings
    latest_data = df.iloc[-1].to_dict()

    st.markdown("---")
    st.subheader("📊 Current Sensor Readings")
    # Display key metrics using gauge charts
    gauge_cols = st.columns(4)

    # Helper to create a gauge chart
    def create_gauge(title, value, max_value, suffix, color='green', threshold=None):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            title={'text': title, 'font': {'size': 18, 'color': 'white'}}, # Adjusted font color for dark mode
            gauge={
                'axis': {'range': [None, max_value], 'tickwidth': 1, 'tickcolor': "white"}, # Adjusted tick color
                'bar': {'color': color},
                'bgcolor': "#262730", # Streamlit dark mode background
                'borderwidth': 2,
                'bordercolor': "#6C757D", # Adjusted border color
                'steps': [
                    {'range': [0, max_value * 0.3], 'color': "rgba(255,255,255,0.1)"}, # Lighter shades for dark mode
                    {'range': [max_value * 0.3, max_value * 0.7], 'color': "rgba(255,255,255,0.2)"},
                    {'range': [max_value * 0.7, max_value], 'color': "rgba(255,255,255,0.3)"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': threshold if threshold is not None else value # If no threshold, just show current value
                }
            }
        ))
        fig.update_layout(height=250, margin=dict(l=10, r=10, t=50, b=10), font={'color': "white", 'family': "Inter"}) # Consistent font and color
        return fig

    # Soil Moisture Gauge
    soil_moisture_val = latest_data.get('soil_moisture')
    if soil_moisture_val is not None and not pd.isna(soil_moisture_val):
        with gauge_cols[0]:
            st.plotly_chart(create_gauge("Soil Moisture (%)", soil_moisture_val, 100, "%", 'rgba(0,170,0,0.8)'), use_container_width=True) # Brighter green
    else:
        with gauge_cols[0]: st.info("Soil Moisture N/A")

    # Temperature Gauge
    temp_val = latest_data.get('temperature')
    if temp_val is not None and not pd.isna(temp_val):
        with gauge_cols[1]:
            st.plotly_chart(create_gauge("Temperature (°C)", temp_val, 40, "°C", 'rgba(255,140,0,0.8)'), use_container_width=True) # Brighter orange
    else:
        with gauge_cols[1]: st.info("Temperature N/A")

    # pH Gauge
    ph_val = latest_data.get('ph') # Use 'ph' after processing
    if ph_val is not None and not pd.isna(ph_val):
        with gauge_cols[2]:
            st.plotly_chart(create_gauge("pH", ph_val, 14, "", 'rgba(65,105,225,0.8)'), use_container_width=True) # Royal blue, better than dark blue
    else:
        with gauge_cols[2]: st.info("pH N/A")

    # Humidity Gauge
    humidity_val = latest_data.get('humidity')
    if humidity_val is not None and not pd.isna(humidity_val):
        with gauge_cols[3]:
            st.plotly_chart(create_gauge("Humidity (%)", humidity_val, 100, "%", 'rgba(147,112,219,0.8)'), use_container_width=True) # MediumPurple
    else:
        with gauge_cols[3]: st.info("Humidity N/A")

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
                    margin=dict(l=40, r=40, t=60, b=40), # Adjust margins
                    plot_bgcolor="#262730", # Match Streamlit dark background
                    paper_bgcolor="#262730" # Match Streamlit dark background
                )
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
            care_tips = crop_care_advice(df, selected_crop_type, lang=voice_lang) # Pass selected language
            st.markdown("---")
            for tip in care_tips:
                st.write(tip)
            
            if st.button(f"🔊 Play Top Alerts ({selected_lang_name})"): # Use display name for button
                if care_tips:
                    for i, tip in enumerate(care_tips[:2]): # Play up to 2 alerts
                        # Remove markdown for better speech, and also remove emojis
                        clean_tip = tip.replace('**', '').replace('🌱', '').replace('💧', '').replace('🌡️', '').replace('💨', '').replace('🧪', '').replace('☀️', '').replace('🌧️', '').replace('✅', '').strip()
                        if clean_tip: # Only play if there's actual text after cleaning
                            st.info(f"Playing alert {i+1}: {clean_tip}")
                            speak_tip(clean_tip, lang=voice_lang)
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
                seed_recommendation = recommend_seeds(current_ph, current_temp, current_rainfall, soil_moisture_pred, lang=voice_lang) # Pass selected language
                st.write(seed_recommendation)
            else:
                st.info("Missing essential current sensor data (pH, temperature, rainfall) for crop suggestions.")
        else:
            st.info("Predicted soil moisture is out of typical range or not available, hindering specific crop suggestions.")

    # --- Real-Time Plant Monitoring (Simulated) ---
    st.subheader("🌿 Real-Time Plant Monitoring (Simulated)")
    try:
        # Fetching directly from Firebase for real-time monitoring
        camera_ref = db.reference('camera_feed/farm1')
        camera_snapshot = camera_ref.order_by_key().limit_to_last(1).get() # Get the very last entry
        
        if camera_snapshot:
            # The snapshot might be a dict with a single key (the push ID)
            latest_event_key = list(camera_snapshot.keys())[0]
            simulated_data = camera_snapshot[latest_event_key]
            st.write(f"🕒 Timestamp: {simulated_data.get('timestamp', 'N/A')}")
            st.success(f"📈 Growth Stage: {simulated_data.get('stage', 'N/A')}")
            st.warning(f"⚠️ Advisory: {simulated_data.get('alert', 'N/A')}")
        else:
            st.info("No simulated camera data found in Firebase. Run `dummy_camera_simulator.py` to start simulation.")
    except Exception as e:
        st.error(f"❌ Error fetching simulated growth data from Firebase: {e}")
        st.info("Ensure `dummy_camera_simulator.py` is running and Firebase rules allow reads.")


    st.markdown("---")
    st.subheader("📋 Latest Sensor Readings (Raw Data)")
    if not df.empty:
        st.dataframe(df.tail(10))
    else:
        st.info("No sensor data to display.")
    
    st.markdown("---")
    # Display initialization messages at the bottom
    with st.expander("Initialization Status"):
        for msg_obj in initialization_messages:
            msg_type = msg_obj["type"]
            msg_content = msg_obj["message"]
            if msg_type == "success":
                st.success(msg_content)
            elif msg_type == "error":
                st.error(msg_content)
            elif msg_type == "warning":
                st.warning(msg_content)
            elif msg_type == "info":
                st.info(msg_content)

    # Auto-refresh every 10 seconds
    st_autorefresh(interval=10 * 1000, key="growth_sim_refresh")
