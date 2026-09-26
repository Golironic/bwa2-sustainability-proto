import json
import pandas as pd
import xgboost as xgb
from pathlib import Path

# Absolute path relative to model_caller.py
ML_DIR = Path(__file__).resolve().parent
MODEL_PATH = ML_DIR / "aqi_xgboost_model.json"
CATEGORIES_PATH = ML_DIR / "region_categories.json"

class AQIPredictor:
    def __init__(
        self, 
        model_path=ML_DIR / "aqi_xgboost_model.json", 
        categories_path=ML_DIR / "region_categories.json"
    ):
        # Convert Path objects to strings for XGBoost/open()
        self.model_path = str(model_path)
        self.categories_path = str(categories_path)

        # Load Model
        self.model = xgb.XGBRegressor()
        self.model.load_model(str(self.model_path))

        # Load Regional Metadata
        with open(self.categories_path, "r") as f:
            self.region_categories = json.load(f)

def load_inference_pipeline():
    """Load the pre-trained model and categorical metadata into memory."""
    # Load model
    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)
    
    # Load saved region categories
    with open(CATEGORIES_PATH, "r") as f:
        categories = json.load(f)
        
    return model, categories

def predict_aqi(model, categories, input_data: dict) -> float:
    """Pass input dictionary to model and return predicted AQI."""
    df_input = pd.DataFrame([input_data])
    
    # Convert 'region' string to matching Categorical type
    df_input['region'] = pd.Categorical(df_input['region'], categories=categories)
    
    # Predict
    prediction = model.predict(df_input)
    return float(prediction[0])

# --- Run Fast Inference ---
if __name__ == "__main__":
    # 1. Load trained resources once
    model, categories = load_inference_pipeline()
    
    # 2. Define real-time input features (from user upload / live API)
    new_sample = {
        'hour': 18,
        'dayofweek': 24,
        'month': 8,
        'temperature_aqi': 31.0,
        'humidity_aqi': 57.0,
        'region': "Gandhinagar",
        'wind_speed_aqi': 11,
        'wind_direction_aqi': 225.0,
    }
    
    # 3. Predict immediately
    predicted_val = predict_aqi(model, categories, new_sample)
    print(f"🔥 Predicted AQI: {predicted_val:.2f}")