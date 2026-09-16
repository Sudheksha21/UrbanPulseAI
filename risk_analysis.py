import pandas as pd
import joblib


# ============================================================
# 1. LOAD TRAINED MODEL
# ============================================================

model_package = joblib.load("models/traffic_model.pkl")

model = model_package["model"]
preprocessor = model_package["preprocessor"]


# ============================================================
# 2. CREATE INPUT DATA
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
# 3. PREDICT TRAFFIC
# ============================================================

input_processed = preprocessor.transform(input_data)

predicted_traffic = model.predict(input_processed)[0]

predicted_traffic = round(predicted_traffic)


# ============================================================
# 4. CALCULATE TRAFFIC RISK
# ============================================================

if predicted_traffic < 2500:

    risk = "LOW"

    recommendation = (
        "Traffic conditions are expected to be light. "
        "Normal transportation capacity should be sufficient."
    )

elif predicted_traffic < 4500:

    risk = "MODERATE"

    recommendation = (
        "Moderate traffic is expected. "
        "Monitor traffic conditions and consider alternative routes."
    )

else:

    risk = "HIGH"

    recommendation = (
        "Heavy traffic is expected. "
        "Consider additional public transport capacity "
        "and alternative routes during peak periods."
    )


# ============================================================
# 5. DISPLAY URBANPULSE RESULT
# ============================================================

print("===== URBANPULSE AI TRAFFIC INTELLIGENCE =====")

print(
    "Predicted Traffic:",
    predicted_traffic,
    "vehicles"
)

print("Risk Level:", risk)

print("Recommendation:", recommendation)