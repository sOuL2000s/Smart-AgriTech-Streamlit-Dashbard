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
import os
import base64
import tempfile

# --- Secure Firebase Initialization ---
if not firebase_admin._apps:
    firebase_key_b64 = os.getenv("FIREBASE_KEY_B64")  # Set in Render Environment
    if not firebase_key_b64:
        st.error("❌ Firebase credentials not found. Please set FIREBASE_KEY_B64 in environment variables.")
    else:
        firebase_json = base64.b64decode(firebase_key_b64).decode("utf-8")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as f:
            f.write(firebase_json.encode())
            f.flush()
            cred = credentials.Certificate(f.name)
            firebase_admin.initialize_app(cred, {
                'databaseURL': 'https://agriastrax-website-default-rtdb.firebaseio.com/'
            })

# --- Load AI Model ---
model_path = "tdann_pnsm_model.keras"
model = tf.keras.models.load_model(model_path)
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
def predict_growth(df):
    features = ['N', 'P', 'K', 'ph', 'rainfall', 'temperature', 'humidity']
    if not all(col in df.columns for col in features):
        return None
    data = df[features].copy()
    scaled = scaler.fit_transform(data)
    lookback = 5
    if len(scaled) < lookback:
        return None
    X = np.array([scaled[-lookback:]])
    prediction = model.predict(X)
    return round(prediction[0][0], 2)

# --- Crop Care Advice ---
def crop_care_advice(df, crop_type):
    latest = df.iloc[-1]
    tips = []

    st.markdown(f"📊 **Latest Data**: {latest.to_dict()}")

    if crop_type.lower() == 'wheat':
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

    elif crop_type.lower() == 'rice':
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

    return tips if tips else ["✅ All parameters look good! Keep monitoring regularly."]

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("🌿 Smart AgriTech Dashboard")

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
        st.subheader("🤖 AI-Based Growth Prediction")
        pred = predict_growth(df)
        if pred:
            st.success(f"📊 Predicted Soil Moisture: {round(pred, 2)}%")
        else:
            st.info("Not enough data to make prediction.")

        st.subheader("🌿 Crop Care Recommendations")
        crop_type = st.selectbox("Select Growing Crop", ["Wheat", "Rice", "Maize", "Millet", "Sorghum"])
        if not df.empty:
            care_tips = crop_care_advice(df, crop_type)
            for tip in care_tips:
                st.write(tip)

        st.subheader("🌾 Crop Suggestion")
        if pred and pred < 30:
            st.write("⚠️ Recommend: **Drought-resistant crops** like Millet or Sorghum")
        elif pred and pred > 70:
            st.write("💧 Recommend: **Water-rich crops** like Rice or Sugarcane")
        else:
            st.write("✅ Recommend: **Balanced crops** like Wheat or Maize")

    st.subheader("📋 Latest Sensor Readings")
    st.dataframe(df.tail(10))
