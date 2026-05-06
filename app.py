import streamlit as st
import joblib
import pandas as pd
import numpy as np
import datetime
import time
import folium
from folium.plugins import Geocoder, MousePosition
from streamlit_folium import st_folium
import plotly.graph_objects as go
from live_weather_api import get_live_weather

# --- Page Configuration ---
st.set_page_config(page_title="MarineSafe Enterprise", page_icon="🌐", layout="wide", initial_sidebar_state="expanded")

# --- Futuristic Custom CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@400;600;700&display=swap');

    .stApp {
        background-color: #050510;
        background-image: 
            radial-gradient(circle at 15% 50%, rgba(6, 182, 212, 0.05), transparent 25%),
            radial-gradient(circle at 85% 30%, rgba(139, 92, 246, 0.05), transparent 25%);
        color: #e2e8f0;
        font-family: 'Rajdhani', sans-serif;
    }
    
    h1, h2, h3, h4 { font-family: 'Orbitron', sans-serif; text-transform: uppercase; }
    
    .title-gradient {
        font-weight: 900;
        background: linear-gradient(90deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0px 0px 10px rgba(0, 242, 254, 0.3);
    }
    
    .metric-card {
        background: rgba(15, 23, 42, 0.8);
        backdrop-filter: blur(15px);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid rgba(6, 182, 212, 0.3);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
    }
    
    .sidebar-card {
        background: rgba(30, 41, 59, 0.6);
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        border-left: 4px solid #06b6d4;
    }

    .stButton > button {
        background: linear-gradient(90deg, #ef4444 0%, #dc2626 100%);
        color: white !important;
        font-family: 'Orbitron', sans-serif;
        font-weight: 700;
        border: none;
        border-radius: 8px;
        padding: 10px 0;
        box-shadow: 0 4px 15px rgba(239, 68, 68, 0.4);
        width: 100%;
        transition: transform 0.2s ease;
    }
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 20px rgba(239, 68, 68, 0.6);
    }
    
    hr { border-top: 1px solid rgba(255, 255, 255, 0.1); margin: 20px 0; }
    </style>
""", unsafe_allow_html=True)

# --- Load Model & Scaler ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('xgboost_model.pkl')
        scaler = joblib.load('scaler.pkl')
        num_cols = joblib.load('num_cols.pkl')
        feature_columns = joblib.load('feature_columns.pkl')
        return model, scaler, num_cols, feature_columns
    except FileNotFoundError:
        return None, None, None, None

model, scaler, num_cols, feature_columns = load_assets()

if model is None:
    st.error("Model assets not found! Please run the training pipeline first.")
    st.stop()

# --- Global State Initialization ---
if "target_lat" not in st.session_state:
    st.session_state.target_lat = 40.71  # Default NY
if "target_lon" not in st.session_state:
    st.session_state.target_lon = -74.00
if "zoom_level" not in st.session_state:
    st.session_state.zoom_level = 3

# --- Sidebar Controls ---
with st.sidebar:
    st.markdown("<h2 class='title-gradient'>COMMAND CENTER</h2>", unsafe_allow_html=True)
    st.markdown("Search for a location using the magnifying glass on the map, or click anywhere to target.")
    
    st.markdown("<div class='sidebar-card'>", unsafe_allow_html=True)
    st.markdown("#### 📍 TARGET COORDINATES")
    st.markdown(f"<h3 style='color: #ef4444; margin:0;'>LAT: {st.session_state.target_lat:.4f}</h3>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='color: #06b6d4; margin:0;'>LON: {st.session_state.target_lon:.4f}</h3>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='sidebar-card'>", unsafe_allow_html=True)
    st.markdown("#### 🛰️ TELEMETRY UPLINK")
    if st.button("📡 FETCH LIVE DATA"):
        with st.spinner("Connecting to Satellites..."):
            time.sleep(0.5)
            st.session_state.live_data = get_live_weather(st.session_state.target_lat, st.session_state.target_lon)
    st.markdown("</div>", unsafe_allow_html=True)

# --- Main Dashboard ---
st.markdown("<h1 class='title-gradient'>MARINESAFE AI ENTERPRISE</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='color: #06b6d4;'>Global Maritime Safety & Predictive Analytics Dashboard</h4>", unsafe_allow_html=True)

# 1. Interactive Map Section
st.markdown("### 🗺️ GLOBAL TACTICAL MAP")
st.markdown("<span style='color:#94a3b8;'>Use the **Search Icon (🔍)** on the left of the map to find any city/port, or simply click anywhere.</span>", unsafe_allow_html=True)

# Create Folium Map centered at the target
# We recreate it to move the red marker, but without manual st.rerun() so it only renders once per click.
m = folium.Map(location=[st.session_state.target_lat, st.session_state.target_lon], zoom_start=st.session_state.zoom_level, tiles="CartoDB dark_matter", control_scale=True)

# Add Geocoder (Search Box)
Geocoder(position='topleft', add_marker=False).add_to(m)

# Add Mouse Position (shows coordinates under cursor)
MousePosition(position='topright', separator=' | ', empty_string='NaN', lng_first=False, num_digits=4).add_to(m)

# Add the red targeting marker
folium.Marker(
    [st.session_state.target_lat, st.session_state.target_lon], 
    popup="Active Target",
    icon=folium.Icon(color="red", icon="crosshairs", prefix="fa")
).add_to(m)

# Render map (st_folium automatically handles clicks and triggers a rerun by itself!)
map_data = st_folium(m, height=450, width="stretch", returned_objects=["last_clicked", "zoom"])

# Update session state for the NEXT run
if map_data:
    if map_data.get("zoom"):
        st.session_state.zoom_level = map_data["zoom"]
        
    if map_data.get("last_clicked"):
        st.session_state.target_lat = map_data["last_clicked"]["lat"]
        st.session_state.target_lon = map_data["last_clicked"]["lng"]

st.markdown("<hr>", unsafe_allow_html=True)

# --- Live Data Display & Prediction Analytics ---
if "live_data" in st.session_state and st.session_state.live_data is not None:
    live_data = st.session_state.live_data
    
    # --- Feature Processing ---
    def process_live_inputs(data_dict):
        dt = datetime.datetime.now()
        hour = dt.hour
        month = dt.month
        
        input_dict = {
            'temp_c': data_dict['temp_c'],
            'wind_speed_knots': data_dict['wind_speed_knots'],
            'wave_height_m': data_dict['wave_height_m'],
            'visibility_km': data_dict['visibility_km'],
            'hour_sin': np.sin(2 * np.pi * hour / 24.0),
            'hour_cos': np.cos(2 * np.pi * hour / 24.0),
            'month_sin': np.sin(2 * np.pi * month / 12.0),
            'month_cos': np.cos(2 * np.pi * month / 12.0)
        }
        
        for col in feature_columns:
            if col not in input_dict:
                if 'temp_c' in col: input_dict[col] = data_dict['temp_c']
                elif 'wind_speed' in col: input_dict[col] = data_dict['wind_speed_knots']
                elif 'wave_height' in col: input_dict[col] = data_dict['wave_height_m']
                elif 'visibility' in col: input_dict[col] = data_dict['visibility_km']
                else: input_dict[col] = 0.0

        df_input = pd.DataFrame([input_dict])[feature_columns]
        df_input_scaled = df_input.copy()
        df_input_scaled[num_cols] = scaler.transform(df_input[num_cols])
        return df_input_scaled

    # Prediction
    processed_input = process_live_inputs(live_data)
    prediction = model.predict(processed_input)[0]
    risk_score = min(99.9, max(1.0, (prediction / 3.0) * 100))
    
    # Determine UI colors
    if risk_score < 30:
        bar_color = "#10b981"
        status = "🟢 SAFE TO NAVIGATE"
        advice = "Conditions are optimal. General navigation approved for all vessel classes. No immediate threats detected by AI."
    elif risk_score < 70:
        bar_color = "#f59e0b"
        status = "🟡 ELEVATED RISK"
        advice = "Caution advised. Small and medium vessels should reconsider routes. Enhanced monitoring protocols activated."
    else:
        bar_color = "#ef4444"
        status = "🔴 SEVERE DANGER"
        advice = "CRITICAL ALERT: Port operations should be halted. Immediate emergency protocols must be engaged. High probability of incidents."

    st.markdown("### 📈 AI PREDICTIVE ANALYTICS")
    col_gauge, col_radar, col_metrics = st.columns([1, 1, 1])

    with col_gauge:
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = risk_score,
            number = {'suffix': "%", 'font': {'color': bar_color, 'size': 50, 'family': 'Orbitron'}},
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "CRITICAL RISK INDEX", 'font': {'color': '#00f2fe', 'size': 20, 'family': 'Orbitron'}},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': bar_color, 'thickness': 0.3},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 2,
                'bordercolor': "rgba(6, 182, 212, 0.3)",
                'steps': [
                    {'range': [0, 30], 'color': 'rgba(16, 185, 129, 0.1)'},
                    {'range': [30, 70], 'color': 'rgba(245, 158, 11, 0.1)'},
                    {'range': [70, 100], 'color': 'rgba(239, 68, 68, 0.1)'}],
            }
        ))
        fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"}, margin=dict(l=20, r=20, t=50, b=20), height=350)
        st.plotly_chart(fig_gauge)

    with col_radar:
        categories = ['Wave Severity', 'Wind Force', 'Visibility Loss', 'Temp Extremes']
        w_score = min(100, live_data['wave_height_m'] * 15)
        wind_score = min(100, live_data['wind_speed_knots'] * 2)
        vis_score = min(100, (20 - min(20, live_data['visibility_km'])) * 5)
        temp_score = min(100, abs(20 - live_data['temp_c']) * 4)
        
        fig_radar = go.Figure(data=go.Scatterpolar(
          r=[w_score, wind_score, vis_score, temp_score],
          theta=categories,
          fill='toself',
          fillcolor='rgba(6, 182, 212, 0.2)',
          line=dict(color='#00f2fe', width=2)
        ))
        fig_radar.update_layout(
          polar=dict(
            radialaxis=dict(visible=False, range=[0, 100]),
            bgcolor='rgba(15, 23, 42, 0.5)'
          ),
          paper_bgcolor="rgba(0,0,0,0)",
          font={'color': "#cbd5e1", 'family': 'Orbitron', 'size': 12},
          title={'text': "ENVIRONMENTAL STRESS PROFILE", 'x': 0.5, 'font': {'color': '#00f2fe', 'size': 18}},
          margin=dict(l=40, r=40, t=50, b=20),
          height=350
        )
        st.plotly_chart(fig_radar)

    with col_metrics:
        st.markdown("<div class='metric-card' style='height: 350px;'>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='color: {bar_color};'>{status}</h4>", unsafe_allow_html=True)
        st.markdown("<hr style='margin: 10px 0;'>", unsafe_allow_html=True)
        
        st.markdown(f"<div style='text-align: left; padding-left: 10px;'>", unsafe_allow_html=True)
        st.markdown(f"**🌊 Wave Height:** <span style='color:#00f2fe;'>{live_data['wave_height_m']:.2f} m</span>", unsafe_allow_html=True)
        st.markdown(f"**🌪️ Wind Speed:** <span style='color:#00f2fe;'>{live_data['wind_speed_knots']:.1f} kts</span>", unsafe_allow_html=True)
        st.markdown(f"**🌫️ Visibility:** <span style='color:#00f2fe;'>{live_data['visibility_km']:.1f} km</span>", unsafe_allow_html=True)
        st.markdown(f"**🌡️ Surface Temp:** <span style='color:#00f2fe;'>{live_data['temp_c']:.1f} °C</span>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<hr style='margin: 10px 0;'>", unsafe_allow_html=True)
        st.markdown(f"**🚨 AI ADVISORY COMMAND:**<br><span style='font-size: 14px; color: #94a3b8;'>{advice}</span>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("Awaiting telemetry request. Please select a target on the map and initiate fetch sequence.")
