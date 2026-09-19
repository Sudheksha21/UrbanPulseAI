import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

from cause_analysis import analyze_causes
import folium
from streamlit_folium import st_folium
from folium.plugins import Geocoder
# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="UrbanPulseAI",
    page_icon="🚦",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model_package = joblib.load(
        "traffic_model.pkl"
    )

    model = model_package["model"]
    preprocessor = model_package["preprocessor"]

    return model, preprocessor


model, preprocessor = load_model()


# ============================================================
# TITLE
# ============================================================

st.title("🚦 UrbanPulseAI")

st.subheader(
    "AI-Based Urban Traffic Prediction and Risk Analysis"
)

st.write(
    "Enter weather and time information to predict "
    "traffic volume and determine the traffic risk level."
)


# ============================================================
# INPUT SECTION
# ============================================================

st.header("🚗 Traffic Prediction Inputs")

col1, col2, col3 = st.columns(3)


# ============================================================
# WEATHER INPUT
# ============================================================

with col1:

    st.subheader("🌦️ Weather Information")

    temp = st.number_input(
        "Temperature",
        value=280.0
    )

    rain_1h = st.number_input(
        "Rain (1 hour)",
        min_value=0.0,
        value=0.0
    )

    snow_1h = st.number_input(
        "Snow (1 hour)",
        min_value=0.0,
        value=0.0
    )

    clouds_all = st.slider(
        "Cloud Coverage",
        0,
        100,
        40
    )


# ============================================================
# TIME INPUT
# ============================================================

with col2:

    st.subheader("🕐 Time Information")

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
        5
    )

    year = st.number_input(
        "Year",
        2000,
        2030,
        2016
    )


# ============================================================
# ADDITIONAL INFORMATION
# ============================================================

with col3:

    st.subheader("📅 Additional Information")

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
        17
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
# FEATURE ENGINEERING
# ============================================================

is_weekend = (
    1 if day_of_week >= 5 else 0
)

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
# BUTTON
# ============================================================

st.markdown("---")

predict_button = st.button(
    "🚦 Predict Traffic",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    # --------------------------------------------------------
    # CREATE INPUT DATA
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # PREPROCESS
    # --------------------------------------------------------

    input_processed = preprocessor.transform(
        input_data
    )


    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    prediction = model.predict(
        input_processed
    )[0]

    predicted_traffic = round(
        prediction
    )


    # ========================================================
    # RISK
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
    # DASHBOARD
    # ========================================================

    st.markdown("---")

    st.header("📊 Urban Traffic Dashboard")

    # FOUR KPI CARDS

    k1, k2, k3, k4 = st.columns(4)


    # CARD 1

    with k1:

        st.metric(
            label="🚗 Predicted Traffic",
            value=f"{predicted_traffic:,} vehicles"
        )


    # CARD 2

    with k2:

        st.metric(
            label="🚨 Risk Level",
            value=risk
        )


    # CARD 3

    with k3:

        if is_rush_hour == 1:

            traffic_period = "Peak Hour"

        else:

            traffic_period = "Normal Hour"


        st.metric(
            label="🕐 Traffic Period",
            value=traffic_period
        )


    # CARD 4

    with k4:

        st.metric(
            label="🌦️ Weather",
            value=weather_main
        )


    # ========================================================
    # GAUGE
    # ========================================================

    st.markdown("---")

    st.subheader("🚦 Traffic Intensity")


    gauge_max = max(
        7000,
        predicted_traffic + 1000
    )


    fig = go.Figure(

        go.Indicator(

            mode="gauge+number",

            value=predicted_traffic,

            title={
                "text": "Predicted Traffic Volume"
            },

            gauge={

                "axis": {
                    "range": [0, gauge_max]
                },

                "steps": [

                    {
                        "range": [
                            0,
                            2500
                        ]
                    },

                    {
                        "range": [
                            2500,
                            4500
                        ]
                    },

                    {
                        "range": [
                            4500,
                            gauge_max
                        ]
                    }

                ],

                "threshold": {

                    "line": {
                        "width": 4
                    },

                    "value": predicted_traffic

                }

            }

        )

    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # ========================================================
    # RESULT
    # ========================================================

    st.markdown("---")

    st.header("📈 Prediction Result")

    result_col1, result_col2 = st.columns(2)


    with result_col1:

        st.metric(
            "Predicted Traffic",
            f"{predicted_traffic:,} vehicles"
        )


    with result_col2:

        if risk == "LOW":

            st.success(
                "🟢 Risk Level: LOW"
            )

        elif risk == "MODERATE":

            st.warning(
                "🟡 Risk Level: MODERATE"
            )

        else:

            st.error(
                "🔴 Risk Level: HIGH"
            )


    # ========================================================
    # CAUSE ANALYSIS
    # ========================================================

    st.markdown("---")

    st.subheader(
        "🔍 Why is this traffic level predicted?"
    )


    causes = analyze_causes(

        predicted_traffic,

        hour,

        day_of_week,

        rain_1h,

        snow_1h,

        clouds_all

    )


    for cause in causes:

        st.write(
            "🔹 " + cause
        )


    # ========================================================
    # RECOMMENDATION
    # ========================================================

    st.markdown("---")

    st.subheader(
        "💡 AI Recommendation"
    )

    st.info(
        recommendation
    )


    # ========================================================
    # URBAN ACTION
    # ========================================================

    st.subheader(
        "🚦 Suggested Urban Action"
    )


    if risk == "HIGH":

        st.error(
            "🚨 Heavy traffic detected. "
            "Increase public transport availability, "
            "monitor major intersections, and recommend "
            "alternative routes."
        )

    elif risk == "MODERATE":

        st.warning(
            "⚠️ Moderate congestion expected. "
            "Monitor traffic flow and encourage "
            "alternative routes."
        )

    else:

        st.success(
            "✅ Traffic conditions are manageable. "
            "Normal traffic management is sufficient."
        )

# ============================================================
# URBAN TRAFFIC LOCATION MAP
# ============================================================

st.subheader("🗺️ Urban Traffic Location Map")

st.write(
    "Search for a city or location to view it on the map."
)

traffic_map = folium.Map(
    location=[20.5937, 78.9629],
    zoom_start=5
)

# Add location search box
Geocoder(
    collapsed=False,
    position="topright",
    add_marker=True
).add_to(traffic_map)

st_folium(
    traffic_map,
    width=700,
    height=450
)
# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "UrbanPulseAI | AI-Based Urban Traffic Intelligence System"
)
