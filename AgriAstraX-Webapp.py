import streamlit as st
import datetime

# Page config
st.set_page_config(page_title="AgriAstraX", layout="wide", initial_sidebar_state="expanded")

# --- Custom CSS for styling ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Orbitron', sans-serif;
        background-color: #0b0c10;
        color: #f5f5f5;
    }

    .sidebar .sidebar-content {
        background-color: #0d1117;
    }

    .big-font {
        font-size:40px !important;
        color: #9eff7d;
    }

    .weather-box {
        background-color: #101820;
        border: 1px solid #9eff7d;
        border-radius: 12px;
        padding: 1em;
        margin-top: 2em;
        display: flex;
        justify-content: space-around;
    }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar Navigation ---
with st.sidebar:
    st.image("https://i.ibb.co/kJykPT2/agri-icon.png", width=60)
    st.markdown("# AgriAstraX")
    st.markdown("### From Family of AstraGenX Corp.")
    st.markdown("---")
    nav = st.radio("", ["Home", "Dashboard", "Crop Scanner", "IoT Control", "Marketplace", "Analytics", "Learning Portal", "Govt Schemes", "Loans", "Settings"])

# --- Main Content ---
if nav == "Home":
    col1, col2 = st.columns([2,3])

    with col1:
        st.markdown("<div class='big-font'>AgriAstraX</div>", unsafe_allow_html=True)
        st.markdown("**Tap To Scan**")
        st.markdown("""
            कृष्णस्य कृपया वृद्धिः च समृद्धिः  
            **AgriAstraX is a Next-Gen Deep Tech marvel — the ultimate single-platform AgriTech solution.**  
            From AI to IoT, revolutionize your farm with precision, intelligence, and seamless control.
        """)

    with col2:
        st.image("https://i.ibb.co/VmVR9RQ/farm-hero.png", use_column_width=True)

    # Weather Section
    with st.container():
        st.markdown("""
        <div class="weather-box">
            <div>🌡️ <strong>24°C</strong> <br>Partly Cloudy</div>
            <div>💧 <strong>65%</strong> <br>Humidity</div>
            <div>🌾 <strong>Recommended:</strong> Winter Wheat</div>
        </div>
        """, unsafe_allow_html=True)

# Optional: Add footer
st.markdown("""
---
<center>AgriAstraX © 2025 | Powered by AstraGenX Corp.</center>
""")
