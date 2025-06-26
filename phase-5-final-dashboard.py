# Phase 5: Unified Streamlit Dashboard for Smart AgriTech

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
        cred = credentials.Certificate("agriastrax-website-firebase-adminsdk-fbsvc-36cdff39c2.json")

    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://agriastrax-website-default-rtdb.firebaseio.com/'
        })
except Exception as e:
    st.error(f"❌ Firebase initialization failed: {e}")

# --- Load Real Crop Labels from CSV ---
crop_df = pd.read_csv("cleaned_sensor_data.csv")
all_crop_labels = sorted(crop_df['label'].unique().tolist())
crop_dummies = pd.get_dummies(pd.Series(all_crop_labels), prefix='crop')

# --- Load AI Model ---
model = tf.keras.models.load_model("tdann_pnsm_model.keras")
scaler = MinMaxScaler()

# --- Fetch Live Sensor Data ---
def fetch_sensor_data():
    ref = db.reference('sensors/farm1')
    snapshot = ref.get()
    if not snapshot:
        return pd.DataFrame()
    df = pd.DataFrame(snapshot).T
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    return df.reset_index(drop=True)

# --- Predict Growth (Soil Moisture Proxy) ---
def predict_growth(df, crop_type):
    base_features = ['N', 'P', 'K', 'ph', 'rainfall', 'temperature', 'humidity']
    if not all(col in df.columns for col in base_features):
        return None
    data = df[base_features].copy()
    scaled = scaler.fit_transform(data)
    lookback = 5
    if len(scaled) < lookback:
        return None
    crop_input = pd.DataFrame([[1 if c.lower().endswith(crop_type.lower()) else 0 for c in crop_dummies.columns]],
                              columns=crop_dummies.columns)
    crop_matrix = np.repeat(crop_input.values, lookback, axis=0)
    full_input = np.hstack((scaled[-lookback:], crop_matrix))
    X = np.array([full_input])
    prediction = model.predict(X)
    return round(prediction[0][0], 2)

# --- Crop Care Advice Function ---
def crop_care_advice(df, crop_type):
    latest = df.iloc[-1]
    tips = []
    st.markdown(f"📊 **Latest Data**: {latest.to_dict()}")

    ct = crop_type.lower()

    if ct == 'wheat':
        if latest['soil_moisture'] < 35:
            tips.append("💧 Irrigate lightly – wheat needs 35–50% soil moisture.")
        if latest['temperature'] > 32:
            tips.append("🌡️ Provide shade or irrigate in evening – temp is too high for wheat.")
        if latest['humidity'] > 70:
            tips.append("💨 Watch out for fungal infections – ensure airflow.")
        if latest['pH'] < 6.0:
            tips.append("🧪 Slightly acidic – consider applying lime to raise pH.")
        if latest['light_intensity'] < 400:
            tips.append("☀️ Light is too low – ensure the crop gets enough sunlight.")

    elif ct == 'rice':
        if latest['soil_moisture'] < 60:
            tips.append("💧 Rice needs high moisture. Ensure proper irrigation.")
        if latest['temperature'] > 38:
            tips.append("🌡️ Too hot for rice. Consider evening irrigation or shade.")
        if latest['humidity'] < 60:
            tips.append("💨 Increase ambient humidity or use mulch.")
        if latest['pH'] < 5.5 or latest['pH'] > 6.5:
            tips.append("🧪 Maintain slightly acidic soil for rice (pH 5.5–6.5).")
        if latest['light_intensity'] < 500:
            tips.append("☀️ Ensure rice gets full sun exposure.")

    elif ct == 'maize':
        if latest['soil_moisture'] < 40:
            tips.append("💧 Maize needs moderate soil moisture levels.")
        if latest['temperature'] < 20:
            tips.append("🌡️ Maize prefers warm weather (20–30°C).")
        if latest['pH'] < 5.8 or latest['pH'] > 7:
            tips.append("🧪 Maintain soil pH between 5.8–7.0.")

    elif ct == 'banana':
        if latest['humidity'] < 60:
            tips.append("💨 Banana requires high humidity. Consider misting or mulching.")
        if latest['soil_moisture'] < 50:
            tips.append("💧 Keep soil consistently moist for banana.")
        if latest['temperature'] < 15:
            tips.append("🌡️ Banana is sensitive to cold – ensure warm conditions.")

    elif ct == 'mango':
        if latest['soil_moisture'] > 60:
            tips.append("💧 Avoid waterlogging. Mango needs well-drained soil.")
        if latest['temperature'] < 20:
            tips.append("🌡️ Mango requires warmer temperatures (>20°C).")

    elif ct == 'grapes':
        if latest['soil_moisture'] > 50:
            tips.append("💧 Grapes prefer drier soil – avoid overwatering.")
        if latest['humidity'] > 70:
            tips.append("💨 High humidity may lead to fungal infections.")

    elif ct == 'cotton':
        if latest['soil_moisture'] < 30:
            tips.append("💧 Cotton requires moderate moisture during flowering.")
        if latest['temperature'] < 20:
            tips.append("🌡️ Cotton thrives in warm temperatures.")

    elif ct == 'millet' or ct == 'sorghum':
        if latest['soil_moisture'] < 25:
            tips.append("💧 These are drought-resistant crops but still need minimal moisture.")
        if latest['temperature'] < 20:
            tips.append("🌡️ Warm climate is ideal for millet/sorghum.")

    elif ct == 'coffee':
        if latest['humidity'] < 60:
            tips.append("💨 Coffee prefers high humidity.")
        if latest['temperature'] < 18:
            tips.append("🌡️ Coffee thrives in 18–24°C range.")

    elif ct == 'jute':
        if latest['soil_moisture'] < 50:
            tips.append("💧 Jute requires ample moisture during growth.")
        if latest['temperature'] < 25:
            tips.append("🌡️ Jute grows well in 25–30°C.")

    elif ct == 'papaya':
        if latest['temperature'] < 20:
            tips.append("🌡️ Papaya prefers 21–33°C range.")
        if latest['pH'] < 6:
            tips.append("🧪 Slightly acidic to neutral soil is best for papaya.")

    elif ct == 'pomegranate':
        if latest['soil_moisture'] > 50:
            tips.append("💧 Avoid overwatering pomegranate.")
        if latest['temperature'] < 20:
            tips.append("🌡️ Ideal temperature is above 20°C.")

    elif ct == 'musk melon' or ct == 'watermelon':
        if latest['soil_moisture'] < 30:
            tips.append("💧 Melons need consistent watering, especially during fruiting.")
        if latest['temperature'] < 25:
            tips.append("🌡️ Ensure temperature is warm (>25°C).")

    elif ct == 'orange':
        if latest['pH'] < 6 or latest['pH'] > 7.5:
            tips.append("🧪 Ideal soil pH for orange is 6.0–7.5.")
        if latest['humidity'] > 70:
            tips.append("💨 Prune trees to improve airflow and prevent fungal issues.")

    elif ct == 'coconut':
        if latest['soil_moisture'] < 50:
            tips.append("💧 Coconut palms need high moisture levels.")
        if latest['temperature'] < 25:
            tips.append("🌡️ Ideal temperature for coconut is above 25°C.")

    elif ct == 'mothbeans':
        if latest['soil_moisture'] < 25:
            tips.append("💧 Mothbeans are drought-tolerant but need minimal irrigation during flowering.")
        if latest['temperature'] < 22:
            tips.append("🌡️ Temperature should be above 22°C.")

    elif ct == 'mungbean':
        if latest['soil_moisture'] < 30:
            tips.append("💧 Ensure regular irrigation during flowering and pod formation.")
        if latest['temperature'] < 20:
            tips.append("🌡️ Mungbean requires warm conditions for optimal growth.")

    elif ct == 'blackgram':
        if latest['soil_moisture'] < 35:
            tips.append("💧 Maintain moderate moisture especially during flowering.")
        if latest['temperature'] < 18:
            tips.append("🌡️ Ideal temperature range is 25–35°C.")

    elif ct == 'lentil':
        if latest['soil_moisture'] < 25:
            tips.append("💧 Lentils need low to moderate moisture.")
        if latest['temperature'] < 15:
            tips.append("🌡️ Lentils grow well in 18–30°C.")

    return tips if tips else ["✅ All parameters look good! Keep monitoring regularly."]

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("🌿 Smart AgriTech Dashboard")

# --- Load and Display Sensor Data ---
df = fetch_sensor_data()

if df.empty:
    st.warning("No data available from Firebase.")
else:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Sensor Trends")
        plot_features = ['soil_moisture', 'temperature', 'humidity', 'pH', 'light_intensity', 'N', 'P', 'K', 'ph', 'rainfall']
        existing_plot_features = [f for f in plot_features if f in df.columns]
        plot_df = df.dropna(subset=existing_plot_features + ['timestamp'])

        if not plot_df.empty and len(existing_plot_features) > 0:
            fig = px.line(
                plot_df,
                x='timestamp',
                y=existing_plot_features,
                labels={'value': 'Sensor Reading', 'timestamp': 'Time'},
                title="Sensor Readings Over Time"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Not enough complete data available for plotting sensor trends.")

    with col2:
        st.subheader("🌿 Crop Care Recommendations")
        crop_type = st.selectbox("Select Growing Crop", all_crop_labels)
        if df is not None and not df.empty:
            care_tips = crop_care_advice(df, crop_type)
            latest = df.iloc[-1]
            st.write("📊 Latest Data:", latest.to_dict())
            for tip in care_tips:
                st.write(tip)

        st.subheader("🤖 AI-Based Growth Prediction")
        pred = predict_growth(df, crop_type)
        if pred:
            st.success(f"📊 Predicted Soil Moisture: {round(pred, 2)}%")
        else:
            st.info("Not enough data to make prediction.")

        st.subheader("🌾 Crop Suggestion")
        if pred and pred < 30:
            st.write("⚠️ Recommend: **Drought-resistant crops** like Millet or Sorghum")
        elif pred and pred > 70:
            st.write("💧 Recommend: **Water-rich crops** like Rice or Sugarcane")
        else:
            st.write("✅ Recommend: **Balanced crops** like Wheat or Maize")

    st.subheader("📋 Latest Sensor Readings")
    st.dataframe(df.tail(10))