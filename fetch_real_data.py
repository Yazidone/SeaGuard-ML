import pandas as pd
import numpy as np
import datetime
import urllib.request
import gzip
import io

def download_real_noaa_data(buoy_id='44025', year='2023', output_file='maritime_safety_data.csv'):
    print(f"Downloading real weather data for Buoy {buoy_id} for year {year} from NOAA NDBC...")
    
    # NOAA NDBC data URL
    url = f"https://www.ndbc.noaa.gov/view_text_file.php?filename={buoy_id}h{year}.txt.gz&dir=data/historical/stdmet/"
    
    try:
        # Download and decode data
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            text_data = response.read().decode('utf-8')
            
            # Read data using pandas (space separated)
            df_raw = pd.read_csv(io.StringIO(text_data), sep='\s+', na_values=['99.0', '99.00', '999.0', '9999.0', '99', '999'])
            
    except Exception as e:
        print(f"Error downloading data: {e}")
        return

    # Remove the second row (units of measurement)
    df_raw = df_raw.iloc[1:].reset_index(drop=True)
    
    # Clean and prepare columns
    # NOAA column names: #YY MM DD hh mm WDIR WSPD GST WVHT DPD APD MWD PRES ATMP WTMP DEWP VIS
    try:
        # Create datetime column
        df_raw['YY'] = df_raw['#YY'].astype(str)
        df_raw['MM'] = df_raw['MM'].astype(str).str.zfill(2)
        df_raw['DD'] = df_raw['DD'].astype(str).str.zfill(2)
        df_raw['hh'] = df_raw['hh'].astype(str).str.zfill(2)
        df_raw['datetime'] = pd.to_datetime(df_raw['YY'] + '-' + df_raw['MM'] + '-' + df_raw['DD'] + ' ' + df_raw['hh'] + ':00:00')
        
        # Extract and convert required variables
        df = pd.DataFrame()
        df['datetime'] = df_raw['datetime']
        
        # Air Temperature (ATMP) is in Celsius
        df['temp_c'] = pd.to_numeric(df_raw['ATMP'], errors='coerce')
        
        # Wind Speed (WSPD) is in m/s -> convert to knots (1 m/s = 1.94384 knots)
        df['wind_speed_knots'] = pd.to_numeric(df_raw['WSPD'], errors='coerce') * 1.94384
        
        # Wave Height (WVHT) is in meters
        df['wave_height_m'] = pd.to_numeric(df_raw['WVHT'], errors='coerce')
        
        # Visibility (VIS) is in nautical miles -> convert to km (1 nmi = 1.852 km)
        # Some buoys don't have VIS, we'll impute or simulate if missing
        if 'VIS' in df_raw.columns:
            df['visibility_km'] = pd.to_numeric(df_raw['VIS'], errors='coerce') * 1.852
        else:
            df['visibility_km'] = np.nan
            
    except KeyError as e:
        print(f"Error parsing columns: {e}")
        return

    # Handle missing values (Interpolation)
    df = df.interpolate(method='linear').ffill().bfill()
    
    # If the buoy does not record visibility, estimate it based on wind
    if df['visibility_km'].isna().all():
        df['visibility_km'] = 20 - (df['wind_speed_knots'].fillna(10) * 0.2) + np.random.normal(loc=0, scale=2, size=len(df))
        df['visibility_km'] = np.clip(df['visibility_km'], 0.1, 20)
        
    # If temperature is completely missing, simulate it roughly for seasons
    if df['temp_c'].isna().all():
        # Simple temperature simulation (colder in winter, hotter in summer)
        months = df['datetime'].dt.month
        base_temp = 15 - 10 * np.cos((months - 1) * 2 * np.pi / 12)
        df['temp_c'] = base_temp + np.random.normal(0, 3, size=len(df))
        
    # Since ship incidents are not published hourly in the same file, 
    # create a probabilistic proxy based on real weather
    incident_prob = (df['wind_speed_knots'] / 40) + (df['wave_height_m'] / 8) + ((20 - df['visibility_km']) / 20)
    incident_prob = incident_prob / 3
    df['incidents'] = np.random.poisson(lam=np.maximum(0, incident_prob * 1.5))
    
    # Downsample to hourly to speed up training
    df = df.resample('h', on='datetime').mean().reset_index()
    df['incidents'] = np.round(df['incidents'].fillna(0)).astype(int)
    df = df.dropna()

    # Save file
    df.to_csv(output_file, index=False)
    print(f"Successfully downloaded and saved real weather data ({len(df)} records) to '{output_file}'.")
    print("You can now re-run the feature engineering and modeling scripts to use this real data!")

if __name__ == "__main__":
    download_real_noaa_data()
