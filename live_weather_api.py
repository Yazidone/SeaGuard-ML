import urllib.request
import json

def get_live_weather(lat: float, lon: float) -> dict:
    """
    Fetches real-time marine and weather data for a given coordinate using Open-Meteo API.
    Returns a dictionary containing the required features for the ML model.
    """
    try:
        # Weather API (Temperature, Wind Speed in knots, Visibility)
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,wind_speed_10m,visibility&wind_speed_unit=kn"
        
        req_w = urllib.request.Request(weather_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req_w) as response:
            weather_data = json.loads(response.read().decode('utf-8'))
            
        # Marine API (Wave Height)
        marine_url = f"https://marine-api.open-meteo.com/v1/marine?latitude={lat}&longitude={lon}&current=wave_height"
        
        req_m = urllib.request.Request(marine_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req_m) as response:
            marine_data = json.loads(response.read().decode('utf-8'))
            
        current_weather = weather_data.get('current', {})
        current_marine = marine_data.get('current', {})
        
        # Extract variables
        temp_c = current_weather.get('temperature_2m', 25.0)
        wind_speed_knots = current_weather.get('wind_speed_10m', 15.0)
        
        # Visibility is in meters, convert to km
        visibility_m = current_weather.get('visibility', 10000.0)
        visibility_km = visibility_m / 1000.0
        
        wave_height_m = current_marine.get('wave_height', 2.0)
        # If wave height is null (e.g., location is on land), default to 0.1
        if wave_height_m is None:
            wave_height_m = 0.1
            
        return {
            'temp_c': float(temp_c),
            'wind_speed_knots': float(wind_speed_knots),
            'wave_height_m': float(wave_height_m),
            'visibility_km': float(visibility_km)
        }
        
    except Exception as e:
        print(f"Failed to fetch live data: {e}")
        # Return fallback default values if API fails
        return {
            'temp_c': 25.0,
            'wind_speed_knots': 15.0,
            'wave_height_m': 2.0,
            'visibility_km': 10.0
        }

if __name__ == "__main__":
    # Test for Port of New York (Lat: 40.71, Lon: -74.00)
    print(get_live_weather(40.71, -74.00))
