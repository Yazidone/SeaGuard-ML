import pandas as pd
import numpy as np

def perform_feature_engineering(input_file='maritime_safety_data.csv', output_file='processed_maritime_data.csv'):
    # Load the data
    try:
        df = pd.read_csv(input_file)
        df['datetime'] = pd.to_datetime(df['datetime'])
    except FileNotFoundError:
        print(f"Error: {input_file} not found. Please run fetch_real_data.py first.")
        return

    # 1. Temporal variables: extract hour, day, and month
    df['hour'] = df['datetime'].dt.hour
    df['day'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month

    # 2. Cyclical Encoding: To preserve temporal continuity
    # Hours (24 hours)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    
    # Months (12 months)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)

    # Sort temporally to ensure accuracy of subsequent operations
    df = df.sort_values(by='datetime').reset_index(drop=True)

    # 3. Lags: For weather variables and past incidents
    lags = [1, 2, 24]
    features_to_lag = ['temp_c', 'wind_speed_knots', 'wave_height_m', 'visibility_km', 'incidents']
    
    for feature in features_to_lag:
        for lag in lags:
            df[f'{feature}_lag_{lag}'] = df[feature].shift(lag)

    # 4. Rolling Averages: To simulate continuous storm effects
    rolling_windows = [6, 12]
    features_to_roll = ['wave_height_m', 'wind_speed_knots']
    
    for feature in features_to_roll:
        for window in rolling_windows:
            df[f'{feature}_rolling_{window}h'] = df[feature].rolling(window=window).mean()

    # Drop missing values (NaNs) resulting from lags and rolling averages
    df = df.dropna().reset_index(drop=True)

    # Drop original columns no longer needed (keep necessary ones)
    # df = df.drop(columns=['hour', 'month']) 

    # Save processed data
    df.to_csv(output_file, index=False)
    print(f"Data processed successfully and saved to {output_file}")
    print(f"New data shape: {df.shape}")

if __name__ == "__main__":
    perform_feature_engineering()
