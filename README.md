<img width="1797" height="809" alt="image" src="https://github.com/user-attachments/assets/255a81b0-fe83-4069-b0b6-aef09c94c8bb" />

# 🌐 MarineSafe AI (SeaGuard ML)

## 📌 Presentation
**MarineSafe AI** is an advanced, production-ready maritime safety and emergency protocol system. By integrating machine learning with real-time geospatial telemetry, the system acts as an early warning Command Center to predict maritime incident risks based on live environmental conditions.

## 🎯 Objectives
- Transform raw meteorological and maritime data into actionable intelligence.
- Predict the **Risk Probability Score** for vessels and port operations in real-time.
- Provide an Enterprise Dashboard for port authorities, coast guards, and logistics companies to prevent accidents and optimize maritime operations.

## 📁 Project Structure
- `app.py`: The main Streamlit Enterprise Dashboard application (Interactive Map, Plotly analytics).
- `live_weather_api.py`: Telemetry module connecting to Open-Meteo satellites.
- `step1_feature_engineering.py`: Data preprocessing and feature extraction pipeline.
- `step2_modeling.py`: Machine learning training pipeline (XGBoost).
- `step3_shap_analysis.py`: Model explainability using SHAP values.
- `Dockerfile` & `.dockerignore`: Containerization for seamless cloud deployment.
- `requirements.txt`: Python dependencies.

## 🚀 Local Installation & Launch
1. **Clone the repository:**
   ```bash
   git clone https://github.com/Yazidone/SeaGuard-ML.git
   cd SeaGuard-ML
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Enterprise Dashboard:**
   ```bash
   streamlit run app.py
   ```

## ☁️ Deployment (Streamlit Cloud & Docker)
This project is fully Dockerized and cloud-ready.
- **Streamlit Community Cloud:** Simply link your GitHub repository and point the main file path to `app.py`.
- **Docker Deployment (AWS/Azure/GCP):**
  ```bash
  docker build -t marinesafe-ai .
  docker run -p 8501:8501 marinesafe-ai
  ```

## 📦 Main Dependencies
- **Machine Learning:** `xgboost`, `scikit-learn`, `shap`, `pandas`, `numpy`
- **UI & Visualization:** `streamlit`, `plotly`, `folium`, `streamlit-folium`
- **Live APIs:** `requests` (Open-Meteo integration)

## 🧠 Available Models
- **XGBoost Regressor:** Tuned for high-accuracy incident prediction based on complex, non-linear weather patterns (Wind Force, Wave Severity, Visibility Loss, Temperature Extremes, and Temporal Seasonality).

## 📊 Features
- **Global Tactical Map:** Interactive Folium map with click-to-target capabilities and built-in Search (Geocoder).
- **Live Satellite Telemetry:** Real-time environmental data fetching using the Open-Meteo Marine/Weather API.
- **AI Predictive Analytics:** Generates a real-time Critical Risk Index (0-100%) visualized via Plotly Gauge charts.
- **Environmental Stress Profile:** Interactive Radar charts for instant condition monitoring.
- **AI Advisory Command:** Automated, actionable safety recommendations based on current risk levels.

## 📈 Key Results
- Successfully achieved high predictive accuracy (Low RMSE) during the modeling phase.
- Transformed theoretical incident counts into an intuitive, operational Risk Probability Percentage.
- Ensured zero-lag map interaction by decoupling Streamlit re-renders from Folium JavaScript clicks.

## 🙏 Acknowledgements
- **Africa TechUp Tour 2025** for the comprehensive training and continuous support.
- **NOAA (National Oceanic and Atmospheric Administration)** for historical maritime data structure inspiration.
- **Open-Meteo** for providing the reliable, real-time marine weather API.
- **Minnesota Department of Transportation (MnDOT)** and **UCI Machine Learning Repository** for foundational data science methodologies and base datasets.

## 📝 License
This project is distributed under the **MIT License**. See the `LICENSE` file for more information.

---
**Author**  
**Lyazid ASSEABBAB**  
*Data Scientist — Africa TechUp Tour 2025*
