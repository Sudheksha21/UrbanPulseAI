import pandas as pd
import joblib


# ============================================================
# 1. LOAD TRAINED MODEL
# ============================================================

model_package = joblib.load("models/traffic_model.pkl")

model = model_package["model"]
preprocessor = model_package["preprocessor"]

print("===== URBANPULSE AI TRAFFIC PREDICTION =====")
print("Model loaded successfully!")


# ============================================================
# 2. CREATE EXAMPLE INPUT
# ============================================================

input_data = pd.DataFrame([{
    "temp": 280.0,
    "rain_1h": 0.0,
    "snow_1h": 0.0,
    "clouds_all": 40,
    "weather_main": "Clouds",
    "weather_description": "scattered clouds",
    "hour": 17,
    "day_of_week": 0,
    "month": 7,
    "year": 2016,
    "is_weekend": 0,
    "day_of_month": 11,
    "week_of_year": 28,
    "is_rush_hour": 1,
    "is_morning_peak": 0,
    "is_evening_peak": 1
}])


# ============================================================
# 3. TRANSFORM INPUT
# ============================================================

input_processed = preprocessor.transform(input_data)


# ============================================================
# 4. PREDICT TRAFFIC
# ============================================================

prediction = model.predict(input_processed)[0]


# ============================================================
# 5. DISPLAY RESULT
# ============================================================

print("\n===== PREDICTION RESULT =====")
print("Predicted Traffic Volume:", round(prediction), "vehicles")