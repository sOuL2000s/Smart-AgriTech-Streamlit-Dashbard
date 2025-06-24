import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import firebase_admin
from firebase_admin import credentials, db
import datetime
import plotly.express as px
import base64
import tempfile
import os
import json

# --- Firebase Secure Setup (Render-Compatible) ---
firebase_key_b64 = os.getenv("FIREBASE_KEY_B64")

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

    # ✅ Prevent double initialization
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://agriastrax-website-default-rtdb.firebaseio.com/'
        })
except Exception as e:
    st.error(f"❌ Firebase initialization failed: {e}")

# --- Load Real Crop Labels from CSV ---
# Make sure 'cleaned_sensor_data.csv' exists in the same directory as your Streamlit app
try:
    crop_df = pd.read_csv("cleaned_sensor_data.csv")
    all_crop_labels = sorted(crop_df['label'].unique().tolist())
    crop_dummies = pd.get_dummies(pd.Series(all_crop_labels), prefix='crop')
except FileNotFoundError:
    st.error("❌ 'cleaned_sensor_data.csv' not found. Please ensure it's in the same directory.")
    all_crop_labels = [] # Initialize as empty to prevent errors later
except Exception as e:
    st.error(f"❌ Error loading 'cleaned_sensor_data.csv': {e}")
    all_crop_labels = []

# --- Load AI Model ---
# Ensure 'tdann_pnsm_model.keras' is in the same directory
try:
    model = tf.keras.models.load_model("tdann_pnsm_model.keras")
except Exception as e:
    st.error(f"❌ Error loading AI model: {e}")
    st.stop() # Stop the app if the model cannot be loaded

scaler = MinMaxScaler()

# --- Fetch Live Sensor Data ---
def fetch_sensor_data():
    """Fetches sensor data from Firebase Realtime Database."""
    ref = db.reference('sensors/farm1')
    snapshot = ref.get()
    if not snapshot:
        return pd.DataFrame()
    df = pd.DataFrame(snapshot).T
    
    # Convert relevant columns to numeric, coercing errors
    # Note: 'ph' is listed twice in your plot_features, I'm assuming it's a typo and should be 'pH'
    numeric_cols = ['N', 'P', 'K', 'pH', 'rainfall', 'temperature', 'humidity', 'soil_moisture', 'light_intensity']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp']) # Drop rows where timestamp conversion failed
    df = df.sort_values('timestamp')
    return df.reset_index(drop=True)

# --- Predict Growth (Soil Moisture Proxy) ---
def predict_growth(df, crop_type):
    """
    Predicts soil moisture using the loaded AI model.
    Assumes the model was trained with 'N', 'P', 'K', 'ph', 'rainfall', 'temperature', 'humidity' 
    and one-hot encoded crop type.
    """
    # Using 'pH' consistently here as per your data, if 'ph' is the actual column name in Firebase, adjust.
    base_features = ['N', 'P', 'K', 'pH', 'rainfall', 'temperature', 'humidity']
    
    # Ensure all required features are present and are numeric
    for col in base_features:
        if col not in df.columns or df[col].isnull().all(): # Check if column is missing or entirely NaN
            st.warning(f"Missing or entirely NaN feature '{col}' in sensor data for prediction. Cannot predict.")
            return None
        # Ensure column is numeric, convert if necessary
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with NaN in base features, as they're required for scaling and prediction
    # Only use the last 'lookback' data points for prediction, ensure they are not NaN
    data = df[base_features].dropna()
    
    lookback = 5
    if len(data) < lookback:
        st.info(f"Not enough complete data points ({len(data)} < {lookback}) for AI prediction after dropping NaNs. Need at least {lookback} consecutive non-NaN rows.")
        return None
    
    # Take the last 'lookback' valid entries
    data_for_prediction = data.iloc[-lookback:]

    # Scale the data
    # It's crucial that the scaler is fitted on data representative of the model's training
    # For prediction, we typically transform data using a pre-fitted scaler.
    # If this scaler is only fitted here on current small data, it might not be ideal.
    # Ideally, `scaler` should be saved after training and loaded alongside the model.
    # For this example, we'll continue fitting it on the latest data.
    scaled = scaler.fit_transform(data_for_prediction) # Fit and transform on just the data for prediction
    
    # Create one-hot encoding for the selected crop type
    # Ensure crop_type matches one of the labels used to create crop_dummies
    crop_col_name = f'crop_{crop_type}'
    if crop_col_name not in crop_dummies.columns:
        st.warning(f"Crop type '{crop_type}' not found in model's known crop labels. Available: {', '.join(col.replace('crop_', '') for col in crop_dummies.columns)}.")
        return None

    crop_input_values = [1 if c == crop_col_name else 0 for c in crop_dummies.columns]
    crop_input = pd.DataFrame([crop_input_values], columns=crop_dummies.columns)
    
    crop_matrix = np.repeat(crop_input.values, lookback, axis=0)
    
    full_input = np.hstack((scaled, crop_matrix)) # Use scaled data directly
    X = np.array([full_input])
    
    try:
        prediction = model.predict(X)
        # Ensure prediction is a single float and then round it
        return round(float(prediction[0][0]), 2)
    except Exception as e:
        st.error(f"Error during AI prediction: {e}")
        return None

# --- Crop Care Advice Function ---
def crop_care_advice(df, crop_type):
    """Provides crop-specific care advice based on latest sensor readings."""
    if df.empty:
        return ["No sensor data available to provide advice."]
    
    # Get the latest row as a dictionary
    latest = df.iloc[-1].to_dict() 
    tips = []
    
    # Filter out timestamp and index for cleaner display
    display_data = {k: v for k, v in latest.items() if k not in ['timestamp', 'index']}
    st.markdown(f"📊 **Latest Data**: {', '.join([f'{k}: {v}' for k, v in display_data.items()])}")

    ct = crop_type.lower()

    # Define thresholds for N, P, K based on general knowledge (you might need to refine these)
    # These are very general and should be calibrated with crop-specific optimal ranges.
    # I'm adding examples for N, P, K but you'll need to fill them for each crop.
    npk_advice = {
        'N': {'min': 50, 'max': 150, 'low_msg': "Consider applying nitrogen-rich fertilizer.", 'high_msg': "Excess nitrogen can promote leafy growth over fruit/flower development."},
        'P': {'min': 20, 'max': 60, 'low_msg': "Consider applying phosphorus fertilizer for root development.", 'high_msg': "High phosphorus can lock up other nutrients."},
        'K': {'min': 50, 'max': 200, 'low_msg': "Consider applying potassium fertilizer for overall plant health and fruit quality.", 'high_msg': "Excess potassium can interfere with calcium and magnesium uptake."},
    }

    for nutrient, thresholds in npk_advice.items():
        if nutrient in latest and not pd.isna(latest[nutrient]):
            value = latest[nutrient]
            if value < thresholds['min']:
                tips.append(f"🌱 **{nutrient} is low ({value})**: {thresholds['low_msg']}")
            elif value > thresholds['max']:
                tips.append(f"🌱 **{nutrient} is high ({value})**: {thresholds['high_msg']}")
            else:
                tips.append(f"🌱 **{nutrient} ({value})**: Looks good.")


    # Specific crop advice (ensure keys are present and not NaN before checking values)
    # soil_moisture
    if 'soil_moisture' in latest and not pd.isna(latest['soil_moisture']):
        sm = latest['soil_moisture']
        if ct == 'wheat':
            if sm < 35: tips.append("💧 Irrigate lightly – wheat needs 35–50% soil moisture.")
        elif ct == 'rice':
            if sm < 60: tips.append("💧 Rice needs high moisture. Ensure proper irrigation.")
        elif ct == 'maize':
            if sm < 40: tips.append("💧 Maize needs moderate soil moisture levels.")
        elif ct == 'banana':
            if sm < 50: tips.append("💧 Keep soil consistently moist for banana.")
        elif ct == 'mango':
            if sm > 60: tips.append("💧 Avoid waterlogging. Mango needs well-drained soil.")
        elif ct == 'grapes':
            if sm > 50: tips.append("💧 Grapes prefer drier soil – avoid overwatering.")
        elif ct == 'cotton':
            if sm < 30: tips.append("💧 Cotton requires moderate moisture during flowering.")
        elif ct == 'millet' or ct == 'sorghum':
            if sm < 25: tips.append("💧 These are drought-resistant crops but still need minimal moisture.")
        elif ct == 'jute':
            if sm < 50: tips.append("💧 Jute requires ample moisture during growth.")
        elif ct == 'pomegranate':
            if sm > 50: tips.append("💧 Avoid overwatering pomegranate.")
        elif ct == 'musk melon' or ct == 'watermelon':
            if sm < 30: tips.append("💧 Melons need consistent watering, especially during fruiting.")
        elif ct == 'coconut':
            if sm < 50: tips.append("💧 Coconut palms need high moisture levels.")
        elif ct == 'mothbeans':
            if sm < 25: tips.append("💧 Mothbeans are drought-tolerant but need minimal irrigation during flowering.")
        elif ct == 'mungbean':
            if sm < 30: tips.append("💧 Ensure regular irrigation during flowering and pod formation.")
        elif ct == 'blackgram':
            if sm < 35: tips.append("💧 Maintain moderate moisture especially during flowering.")
        elif ct == 'lentil':
            if sm < 25: tips.append("💧 Lentils need low to moderate moisture.")

    # temperature
    if 'temperature' in latest and not pd.isna(latest['temperature']):
        temp = latest['temperature']
        if ct == 'wheat':
            if temp > 32: tips.append("🌡️ Provide shade or irrigate in evening – temp is too high for wheat.")
        elif ct == 'rice':
            if temp > 38: tips.append("🌡️ Too hot for rice. Consider evening irrigation or shade.")
        elif ct == 'maize':
            if temp < 20: tips.append("🌡️ Maize prefers warm weather (20–30°C).")
        elif ct == 'banana':
            if temp < 15: tips.append("🌡️ Banana is sensitive to cold – ensure warm conditions.")
        elif ct == 'mango':
            if temp < 20: tips.append("🌡️ Mango requires warmer temperatures (>20°C).")
        elif ct == 'cotton':
            if temp < 20: tips.append("🌡️ Cotton thrives in warm temperatures.")
        elif ct == 'millet' or ct == 'sorghum':
            if temp < 20: tips.append("🌡️ Warm climate is ideal for millet/sorghum.")
        elif ct == 'coffee':
            if temp < 18: tips.append("🌡️ Coffee thrives in 18–24°C range.")
        elif ct == 'jute':
            if temp < 25: tips.append("🌡️ Jute grows well in 25–30°C.")
        elif ct == 'papaya':
            if temp < 20: tips.append("🌡️ Papaya prefers 21–33°C range.")
        elif ct == 'pomegranate':
            if temp < 20: tips.append("🌡️ Ideal temperature is above 20°C.")
        elif ct == 'musk melon' or ct == 'watermelon':
            if temp < 25: tips.append("🌡️ Ensure temperature is warm (>25°C).")
        elif ct == 'coconut':
            if temp < 25: tips.append("🌡️ Ideal temperature for coconut is above 25°C.")
        elif ct == 'mothbeans':
            if temp < 22: tips.append("🌡️ Temperature should be above 22°C.")
        elif ct == 'mungbean':
            if temp < 20: tips.append("🌡️ Mungbean requires warm conditions for optimal growth.")
        elif ct == 'blackgram':
            if temp < 18: tips.append("🌡️ Ideal temperature range is 25–35°C.")
        elif ct == 'lentil':
            if temp < 15: tips.append("🌡️ Lentils grow well in 18–30°C.")

    # humidity
    if 'humidity' in latest and not pd.isna(latest['humidity']):
        hum = latest['humidity']
        if ct == 'wheat':
            if hum > 70: tips.append("💨 Watch out for fungal infections – ensure airflow.")
        elif ct == 'rice':
            if hum < 60: tips.append("💨 Increase ambient humidity or use mulch.")
        elif ct == 'banana':
            if hum < 60: tips.append("💨 Banana requires high humidity. Consider misting or mulching.")
        elif ct == 'grapes':
            if hum > 70: tips.append("💨 High humidity may lead to fungal infections.")
        elif ct == 'coffee':
            if hum < 60: tips.append("💨 Coffee prefers high humidity.")
        elif ct == 'orange':
            if hum > 70: tips.append("💨 Prune trees to improve airflow and prevent fungal issues.")

    # pH
    if 'pH' in latest and not pd.isna(latest['pH']):
        ph_val = latest['pH']
        if ct == 'wheat':
            if ph_val < 6.0: tips.append("🧪 Slightly acidic – consider applying lime to raise pH.")
        elif ct == 'rice':
            if ph_val < 5.5 or ph_val > 6.5: tips.append("🧪 Maintain slightly acidic soil for rice (pH 5.5–6.5).")
        elif ct == 'maize':
            if ph_val < 5.8 or ph_val > 7: tips.append("🧪 Maintain soil pH between 5.8–7.0.")
        elif ct == 'papaya':
            if ph_val < 6: tips.append("🧪 Slightly acidic to neutral soil is best for papaya.")
        elif ct == 'orange':
            if ph_val < 6 or ph_val > 7.5: tips.append("🧪 Ideal soil pH for orange is 6.0–7.5.")

    # light_intensity
    if 'light_intensity' in latest and not pd.isna(latest['light_intensity']):
        light = latest['light_intensity']
        if ct == 'wheat':
            if light < 400: tips.append("☀️ Light is too low – ensure the crop gets enough sunlight.")
        elif ct == 'rice':
            if light < 500: tips.append("☀️ Ensure rice gets full sun exposure.")
            
    # Add rainfall advice
    if 'rainfall' in latest and not pd.isna(latest['rainfall']):
        rain = latest['rainfall']
        if rain < 50: # Example threshold for low rainfall
            tips.append("🌧️ Current rainfall is low. Consider supplementary irrigation, especially for water-intensive crops.")
        elif rain > 200: # Example threshold for high rainfall
            tips.append("🌧️ High rainfall. Ensure good drainage to prevent waterlogging and root rot.")

    return tips if tips else ["✅ All major parameters look good! Keep monitoring regularly for optimal growth."]

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("🌿 Smart AgriTech Dashboard")

# --- Load and Display Sensor Data ---
df = fetch_sensor_data()

if df.empty:
    st.warning("No data available from Firebase. Please ensure your sensor sends data or check Firebase connection.")
else:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Sensor Trends")
        # Ensure all columns exist before trying to plot them
        # Corrected: 'ph' was a duplicate, assuming 'pH' is the correct one
        plot_features = ['soil_moisture', 'temperature', 'humidity', 'pH', 'light_intensity', 'N', 'P', 'K', 'rainfall']
        existing_plot_features = [f for f in plot_features if f in df.columns]
        
        # Melt the DataFrame for Plotly Express to handle different value ranges on a single y-axis
        # This converts from wide to long format, which is generally better for time series with multiple variables
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
                    color='Sensor Metric', # Different lines for each metric
                    labels={'Reading': 'Sensor Reading', 'timestamp': 'Time'},
                    title="Sensor Readings Over Time"
                )
                fig.update_layout(hovermode="x unified") # Improves hover experience
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
            care_tips = crop_care_advice(df, selected_crop_type)
            st.markdown("---") # Separator for better readability
            for tip in care_tips:
                st.write(tip)
        elif not selected_crop_type:
            st.info("Please select a crop to get recommendations.")
        else:
            st.info("No sensor data available for crop care recommendations.")

        st.subheader("🤖 AI-Based Growth Prediction")
        if df is not None and not df.empty and selected_crop_type:
            pred = predict_growth(df, selected_crop_type)
            if pred is not None:
                # Add a sanity check for prediction value
                if 0 <= pred <= 100:
                    st.success(f"📊 Predicted Soil Moisture: **{pred:.2f}%**") # Format to 2 decimal places
                else:
                    st.warning(f"📊 Predicted Soil Moisture: **{pred:.2f}%**. This value seems unusual. (Expected between 0-100%).")
            else:
                st.info("Not enough data or issue with model prediction. Check logs above for details.")
        else:
            st.info("Select a crop and ensure sensor data is available for prediction.")

        st.subheader("🌾 Crop Suggestion")
        if df is not None and not df.empty and selected_crop_type and pred is not None and (0 <= pred <= 100):
            if pred < 30:
                st.write("⚠️ Based on predicted low soil moisture, consider **drought-resistant crops** like **Millet** or **Sorghum**.")
            elif pred > 70:
                st.write("💧 Based on predicted high soil moisture, consider **water-rich crops** like **Rice** or **Sugarcane**.")
            else:
                st.write("✅ Based on predicted soil moisture, **balanced crops** like **Wheat** or **Maize** are recommended.")
        else:
            st.info("Prediction needed for crop suggestions or prediction out of typical range.")

    st.markdown("---")
    st.subheader("📋 Latest Sensor Readings")
    if not df.empty:
        st.dataframe(df.tail(10))
    else:
        st.info("No sensor data to display.")