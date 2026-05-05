import pandas as pd
import numpy as np

def generate_mock_data(num_samples=5000):
    np.random.seed(42)
    
    # Create datetime range
    dates = pd.date_range(start='2024-01-01', periods=num_samples, freq='H')
    
    # Generate variables
    temp_c = np.random.normal(loc=20, scale=10, size=num_samples) # Temperature between ~0 and ~40
    wind_speed_knots = np.random.gamma(shape=2, scale=10, size=num_samples) # Wind speed
    wave_height_m = wind_speed_knots * 0.1 + np.random.normal(loc=0.5, scale=0.5, size=num_samples) # Correlated with wind
    wave_height_m = np.clip(wave_height_m, 0.1, 15) # Cap at realistic values
    
    visibility_km = 20 - (wind_speed_knots * 0.2) + np.random.normal(loc=0, scale=2, size=num_samples)
    visibility_km = np.clip(visibility_km, 0.1, 20) # Cap visibility
    
    # Incidents calculation based on weather (higher wind/waves, lower visibility = more incidents)
    incident_prob = (wind_speed_knots / 50) + (wave_height_m / 10) + ( (20 - visibility_km) / 20 )
    incident_prob = incident_prob / 3 # normalize somewhat
    
    # Add some random noise
    incidents = np.random.poisson(lam=incident_prob * 2)
    
    df = pd.DataFrame({
        'datetime': dates,
        'temp_c': temp_c,
        'wind_speed_knots': wind_speed_knots,
        'wave_height_m': wave_height_m,
        'visibility_km': visibility_km,
        'incidents': incidents
    })
    
    df.to_csv('maritime_safety_data.csv', index=False)
    print("Dataset generated successfully at 'maritime_safety_data.csv'.")

if __name__ == "__main__":
    generate_mock_data()
