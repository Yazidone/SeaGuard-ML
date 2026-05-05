import pandas as pd
import numpy as np

def perform_feature_engineering(input_file='maritime_safety_data.csv', output_file='processed_maritime_data.csv'):
    # Load the data
    try:
        df = pd.read_csv(input_file)
        df['datetime'] = pd.to_datetime(df['datetime'])
    except FileNotFoundError:
        print(f"Error: {input_file} not found. Please run generate_data.py first.")
        return

    # 1. متغيرات زمنية: استخراج الساعة، اليوم، والشهر
    df['hour'] = df['datetime'].dt.hour
    df['day'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month

    # 2. ترميز دائري (Cyclical Encoding): للحفاظ على الاستمرارية الزمنية
    # الساعات (24 ساعة)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    
    # الأشهر (12 شهر)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)

    # الترتيب زمنياً لضمان دقة العمليات التالية
    df = df.sort_values(by='datetime').reset_index(drop=True)

    # 3. تأخيرات زمنية (Lags): لمتغيرات الطقس وعدد الحوادث السابقة
    lags = [1, 2, 24]
    features_to_lag = ['temp_c', 'wind_speed_knots', 'wave_height_m', 'visibility_km', 'incidents']
    
    for feature in features_to_lag:
        for lag in lags:
            df[f'{feature}_lag_{lag}'] = df[feature].shift(lag)

    # 4. متوسطات متحركة (Rolling Averages): لمحاكاة تأثير العواصف المستمرة
    rolling_windows = [6, 12]
    features_to_roll = ['wave_height_m', 'wind_speed_knots']
    
    for feature in features_to_roll:
        for window in rolling_windows:
            df[f'{feature}_rolling_{window}h'] = df[feature].rolling(window=window).mean()

    # إسقاط القيم المفقودة (NaNs) الناتجة عن التأخيرات والمتوسطات المتحركة
    df = df.dropna().reset_index(drop=True)

    # إسقاط الأعمدة الأصلية التي لم نعد بحاجة إليها للنمذجة أو إبقائها (نحتفظ بالضروري)
    # df = df.drop(columns=['hour', 'month']) 

    # حفظ البيانات المعالجة
    df.to_csv(output_file, index=False)
    print(f"تمت معالجة البيانات بنجاح وتم حفظها في {output_file}")
    print(f"شكل البيانات الجديد: {df.shape}")

if __name__ == "__main__":
    perform_feature_engineering()
