import streamlit as st
import pandas as pd
import joblib


# ============================================================
# LOAD MODEL
# ============================================================

model_package = joblib.load("models/traffic_model.pkl")

model = model_package["model"]
preprocessor = model_package["preprocessor"]


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="UrbanPulseAI",
    page_icon="🚦",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🚦 UrbanPulseAI")
st.subheader("AI-Based Urban Traffic Prediction and Risk Analysis")

st.write(
    "Enter weather and time information to predict traffic volume "
    "and determine the traffic risk level."
)


# ============================================================
# INPUT SECTION
# ============================================================

st.header("Traffic Prediction Inputs")

col1, col2, col3 = st.columns(3)


with col1:

    temp = st.number_input(
        "Temperature",
        value=280.0
    )

    rain_1h = st.number_input(
        "Rain (1 hour)",
        value=0.0
    )

    snow_1h = st.number_input(
        "Snow (1 hour)",
        value=0.0
    )

    clouds_all = st.slider(
        "Cloud Coverage",
        0,
        100,
        40
    )


with col2:

    hour = st.slider(
        "Hour",
        0,
        23,
        17
    )

    day_of_week = st.slider(
        "Day of Week (0 = Monday)",
        0,
        6,
        0
    )

    month = st.slider(
        "Month",
        1,
        12,
        7
    )

    year = st.number_input(
        "Year",
        min_value=2000,
        max_value=2030,
        value=2016
    )


with col3:

    day_of_month = st.slider(
        "Day of Month",
        1,
        31,
        11
    )

    week_of_year = st.slider(
        "Week of Year",
        1,
        53,
        28
    )

    weather_main = st.selectbox(
        "Weather",
        [
            "Clear",
            "Clouds",
            "Rain",
            "Drizzle",
            "Thunderstorm",
            "Snow",
            "Mist"
        ]
    )

    weather_description = st.text_input(
        "Weather Description",
        "scattered clouds"
    )


# ============================================================
# CALCULATE FEATURES
# ============================================================

is_weekend = 1 if day_of_week >= 5 else 0

is_rush_hour = (
    1 if hour in [7, 8, 9, 16, 17, 18, 19]
    else 0
)

is_morning_peak = (
    1 if hour in [7, 8, 9]
    else 0
)

is_evening_peak = (
    1 if hour in [16, 17, 18, 19]
    else 0
)


# ============================================================
# PREDICTION
# ============================================================

if st.button("🚦 Predict Traffic"):

    input_data = pd.DataFrame([{

        "temp": temp,
        "rain_1h": rain_1h,
        "snow_1h": snow_1h,
        "clouds_all": clouds_all,

        "weather_main": weather_main,
        "weather_description": weather_description,

        "hour": hour,
        "day_of_week": day_of_week,
        "month": month,
        "year": year,

        "is_weekend": is_weekend,
        "day_of_month": day_of_month,
        "week_of_year": week_of_year,

        "is_rush_hour": is_rush_hour,
        "is_morning_peak": is_morning_peak,
        "is_evening_peak": is_evening_peak

    }])


    # Transform data

    input_processed = preprocessor.transform(input_data)


    # Predict

    prediction = model.predict(input_processed)[0]

    predicted_traffic = round(prediction)


    # ========================================================
    # RISK ANALYSIS
    # ========================================================

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


    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    st.header("Prediction Result")

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        st.metric(
            "Predicted Traffic",
            f"{predicted_traffic:,} vehicles"
        )

    with result_col2:

        if risk == "LOW":

            st.success(f"Risk Level: {risk}")

        elif risk == "MODERATE":

            st.warning(f"Risk Level: {risk}")

        else:

            st.error(f"Risk Level: {risk}")


    st.info(
        f"💡 Recommendation: {recommendation}"
    )