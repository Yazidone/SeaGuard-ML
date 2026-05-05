import streamlit as st
import joblib
import pandas as pd
import numpy as np
import datetime

# --- إعدادات الصفحة والتصميم (Tailwind-like CSS) ---
st.set_page_config(page_title="MarineSafe AI", page_icon="🚢", layout="wide")

st.markdown("""
    <style>
    /* Tailwind inspired UI */
    .main {
        background-color: #f8fafc;
    }
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        text-align: center;
        margin-top: 20px;
        transition: all 0.3s ease;
    }
    .safe { border-bottom: 5px solid #10b981; }
    .warning { border-bottom: 5px solid #f59e0b; }
    .danger { border-bottom: 5px solid #ef4444; }
    
    h1 {
        color: #1e293b;
        font-weight: 800;
    }
    h3 {
        color: #475569;
    }
    .stSlider > div > div > div > div {
        background-color: #3b82f6 !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- تحميل النموذج والبيانات ---
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

# --- عنوان التطبيق ---
st.markdown("<h1>🚢 MarineSafe AI <span style='font-size: 20px; color: #64748b;'>by Elyazid Asbab</span></h1>", unsafe_allow_html=True)
st.markdown("<h3>نظام التنبؤ الذكي بحوادث الطوارئ والسلامة البحرية</h3>", unsafe_allow_html=True)
st.markdown("---")

if model is None:
    st.error("لم يتم العثور على النموذج المدرب! يرجى تشغيل سكربتات التهيئة أولاً.")
    st.stop()

# --- واجهة المستخدم (المدخلات) ---
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 🌊 الظروف البحرية والجوية")
    wave_height_m = st.slider("ارتفاع الموج (متر)", min_value=0.1, max_value=15.0, value=2.0, step=0.1)
    wind_speed_knots = st.slider("سرعة الرياح (عقدة)", min_value=0.0, max_value=100.0, value=15.0, step=1.0)
    visibility_km = st.slider("مدى الرؤية (كم)", min_value=0.1, max_value=20.0, value=10.0, step=0.5)
    temp_c = st.slider("درجة الحرارة (مئوية)", min_value=-10.0, max_value=50.0, value=25.0, step=0.5)

with col2:
    st.markdown("#### 🕒 التوقيت")
    input_date = st.date_input("التاريخ", datetime.date.today())
    input_time = st.time_input("الوقت", datetime.datetime.now().time())

# --- معالجة المدخلات وهندسة الميزات ---
def process_inputs():
    # 1. استخراج المتغيرات الزمنية
    dt = datetime.datetime.combine(input_date, input_time)
    hour = dt.hour
    month = dt.month
    
    # 2. الترميز الدائري
    hour_sin = np.sin(2 * np.pi * hour / 24.0)
    hour_cos = np.cos(2 * np.pi * hour / 24.0)
    month_sin = np.sin(2 * np.pi * month / 12.0)
    month_cos = np.cos(2 * np.pi * month / 12.0)
    
    # بناء قاموس المتغيرات
    input_dict = {
        'temp_c': temp_c,
        'wind_speed_knots': wind_speed_knots,
        'wave_height_m': wave_height_m,
        'visibility_km': visibility_km,
        'hour_sin': hour_sin,
        'hour_cos': hour_cos,
        'month_sin': month_sin,
        'month_cos': month_cos
    }
    
    # إضافة قيم افتراضية للتأخيرات والمتوسطات المتحركة (نظراً لأننا لا نملك تسلسلاً للمستخدم)
    # في نظام الإنتاج الحقيقي، ستأتي هذه القيم من قاعدة البيانات لأخر 24 ساعة
    for col in feature_columns:
        if col not in input_dict:
            # وضع قيم افتراضية (مثلاً نفس القيم الحالية أو متوسطات قريبة)
            if 'temp_c' in col: input_dict[col] = temp_c
            elif 'wind_speed' in col: input_dict[col] = wind_speed_knots
            elif 'wave_height' in col: input_dict[col] = wave_height_m
            elif 'visibility' in col: input_dict[col] = visibility_km
            elif 'incidents' in col: input_dict[col] = 0.0 # افتراض 0 حوادث سابقة
            else: input_dict[col] = 0.0

    # تحويل إلى DataFrame بنفس الترتيب
    df_input = pd.DataFrame([input_dict])[feature_columns]
    
    # 3. التوحيد القياسي
    df_input_scaled = df_input.copy()
    df_input_scaled[num_cols] = scaler.transform(df_input[num_cols])
    
    return df_input_scaled

# --- عرض النتيجة ---
st.markdown("---")
if st.button("توقع الحوادث 🚀", use_container_width=True):
    with st.spinner('جاري تحليل البيانات...'):
        processed_input = process_inputs()
        prediction = model.predict(processed_input)[0]
        
        # تصنيف الخطورة
        predicted_incidents = max(0, round(prediction)) # لا يمكن أن يكون سالباً
        
        if predicted_incidents == 0:
            status_class = "safe"
            status_text = "آمن - احتماليات حوادث شبه معدومة"
            color = "#10b981"
        elif predicted_incidents <= 2:
            status_class = "warning"
            status_text = "متوسط - يرجى توخي الحذر"
            color = "#f59e0b"
        else:
            status_class = "danger"
            status_text = "عالي الخطورة - طوارئ محتملة"
            color = "#ef4444"
            
        st.markdown(f"""
        <div class="metric-card {status_class}">
            <h2 style="color: #64748b; margin-bottom: 0;">العدد المتوقع للحوادث في هذه الظروف</h2>
            <h1 style="color: {color}; font-size: 60px; margin: 10px 0;">{predicted_incidents}</h1>
            <h3 style="color: {color};">{status_text}</h3>
        </div>
        """, unsafe_allow_html=True)
