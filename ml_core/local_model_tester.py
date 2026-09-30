import json
import pandas as pd
import xgboost as xgb
from pathlib import Path

try:
    from ml_core.config import CATEGORIES_PATH, MODEL_PATH, FEATURE_COLS
except ImportError:  # run directly from inside ml_core/
    from config import CATEGORIES_PATH, MODEL_PATH, FEATURE_COLS

# 1. Load trained model & regional metadata
model = xgb.XGBRegressor()
model.load_model(str(MODEL_PATH))

with open(CATEGORIES_PATH, "r") as f:
    region_categories = json.load(f)

# 2. Define test scenarios matching the full 19-feature schema
# Columns: ['hour', 'dayofweek', 'month', 'region', 'temperature_aqi', 'humidity_aqi', 
#          'wind_speed_aqi', 'wind_direction_aqi', 'aqi_lag_0h', 'aqi_lag_1h', 'aqi_lag_2h', 
#          'aqi_lag_3h', 'aqi_lag_24h', 'temp_lag_1h', 'wind_lag_1h', 
#          'aqi_roll_mean_3h', 'aqi_roll_mean_6h', 'aqi_roll_mean_24h', 'aqi_roll_std_24h']

test_cases = pd.DataFrame([
    # Case 1: High Pollution Evening (Delhi - Winter/Nov, low wind, high lag AQI)
    {
        'hour': 20, 'dayofweek': 0, 'month': 11, 'region': 'Delhi',
        'temperature_aqi': 18.0, 'humidity_aqi': 75.0, 'wind_speed_aqi': 3.5, 'wind_direction_aqi': 290.0,
        'aqi_lag_0h': 315.0, 'aqi_lag_1h': 310.0, 'aqi_lag_2h': 300.0, 'aqi_lag_3h': 290.0, 'aqi_lag_24h': 280.0,
        'temp_lag_1h': 19.5, 'wind_lag_1h': 4.0,
        'aqi_roll_mean_3h': 300.0, 'aqi_roll_mean_6h': 285.0, 'aqi_roll_mean_24h': 260.0, 'aqi_roll_std_24h': 25.0
    },
    
    # Case 2: Moderate Coastal Afternoon (Mumbai - Summer/May, high sea breeze)
    {
        'hour': 14, 'dayofweek': 2, 'month': 5, 'region': 'Mumbai',
        'temperature_aqi': 33.0, 'humidity_aqi': 68.0, 'wind_speed_aqi': 14.5, 'wind_direction_aqi': 240.0,
        'aqi_lag_0h': 104.0, 'aqi_lag_1h': 105.0, 'aqi_lag_2h': 110.0, 'aqi_lag_3h': 115.0, 'aqi_lag_24h': 120.0,
        'temp_lag_1h': 32.5, 'wind_lag_1h': 13.0,
        'aqi_roll_mean_3h': 110.0, 'aqi_roll_mean_6h': 112.0, 'aqi_roll_mean_24h': 118.0, 'aqi_roll_std_24h': 8.5
    },

    # Case 3: Clear Day (Gandhinagar - Spring/March, moderate weather)
    {
        'hour': 11, 'dayofweek': 4, 'month': 3, 'region': 'Gandhinagar',
        'temperature_aqi': 28.5, 'humidity_aqi': 45.0, 'wind_speed_aqi': 8.0, 'wind_direction_aqi': 180.0,
        'aqi_lag_0h': 87.0, 'aqi_lag_1h': 88.0, 'aqi_lag_2h': 85.0, 'aqi_lag_3h': 82.0, 'aqi_lag_24h': 95.0,
        'temp_lag_1h': 27.0, 'wind_lag_1h': 7.5,
        'aqi_roll_mean_3h': 85.0, 'aqi_roll_mean_6h': 87.0, 'aqi_roll_mean_24h': 92.0, 'aqi_roll_std_24h': 5.2
    }
])

test_cases = test_cases[FEATURE_COLS]

# 3. Format region as categorical matching trained categories
test_cases['region'] = pd.Categorical(test_cases['region'], categories=region_categories)

# 4. Predict
try:
    predictions = model.predict(test_cases)
    
    for i, pred in enumerate(predictions):
        region_name = test_cases.iloc[i]['region']
        print(f"📍 Test Case {i+1} ({region_name}): Predicted AQI = {pred:.2f}")

except Exception as e:
    print(f"❌ Error predicting AQI: {e}")