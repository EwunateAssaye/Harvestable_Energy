import streamlit as st
import numpy as np
import pandas as pd
import joblib


# ------------------------------------
# Load model and feature list
# ------------------------------------
model = joblib.load("xgboost_ghi_model.pkl")
features = joblib.load("model_features.pkl")


# ------------------------------------
# Streamlit UI
# ------------------------------------
st.set_page_config(page_title="Solar Harvestable Energy Prediction", layout="centered")

st.title("🌞 Potential Harvestable Solar Energy Predictor")
st.write(
    "Predict **potential harvestable solar energy (Wh/m² per hour)** "
    "using meteorological and temporal inputs."
)


# ------------------------------------
# User input form
# ------------------------------------
with st.form("ghi_input_form"):
    st.subheader("Input Parameters")

    # Time inputs
    hour = st.slider("Hour of day", 0, 23, 12)
    dayofyear = st.slider("Day of year", 1, 365, 180)

    # Meteorological inputs
    temperature = st.number_input("Air Temperature (°C)", -30.0, 50.0, 20.0)
    humidity = st.number_input("Relative Humidity (%)", 0.0, 100.0, 50.0)
    pressure = st.number_input("Pressure (hPa)", 800.0, 1100.0, 1013.0)
    wind_speed = st.number_input("Wind Speed (m/s)", 0.0, 20.0, 2.0)
    precipitable_water = st.number_input("Precipitable Water (cm)", 0.0, 10.0, 1.5)
    solar_zenith = st.number_input("Solar Zenith Angle (degrees)", 0.0, 180.0, 45.0)

    submit = st.form_submit_button("Estimate Harvestable Energy")


# ------------------------------------
# Prediction logic
# ------------------------------------
if submit:
    # Cyclical time encoding
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)

    doy_sin = np.sin(2 * np.pi * dayofyear / 365)
    doy_cos = np.cos(2 * np.pi * dayofyear / 365)

    # ------------------------------------------------
    # DEFAULT VALUES FOR TRAINING FEATURES
    # ------------------------------------------------
    wind_direction = 180.0          # degrees
    snow_depth = 0.0                # meters
    dew_point = temperature - 5.0   # simple meteorological approximation

    # ------------------------------------------------
    # Assemble input data
    # ------------------------------------------------
    input_data = {
        "Temperature": temperature,
        "Dew Point": dew_point,
        "Relative Humidity": humidity,
        "Pressure": pressure,
        "Wind Speed": wind_speed,
        "Wind Direction": wind_direction,
        "Snow Depth": snow_depth,
        "Precipitable Water": precipitable_water,
        "Solar Zenith Angle": solar_zenith,
        "dayofyear": dayofyear,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "doy_sin": doy_sin,
        "doy_cos": doy_cos
    }

    X_input = pd.DataFrame([input_data])

    # ------------------------------------------------
    # Defensive feature check
    # ------------------------------------------------
    missing_features = set(features) - set(X_input.columns)
    if missing_features:
        st.error(f"Missing required features: {missing_features}")
        st.stop()

    X_input = X_input[features]

    # ------------------------------------------------
    # Predict GHI
    # ------------------------------------------------
    ghi_pred = model.predict(X_input)[0]
    ghi_pred = max(0, ghi_pred)

    # ------------------------------------------------
    # Convert to harvestable energy
    # ------------------------------------------------
    efficiency = 0.16  # 16% PV system efficiency
    harvestable_energy = ghi_pred * efficiency  # Wh/m² per hour

    # ------------------------------------------------
    # Display result
    # ------------------------------------------------
    st.success(
        f"🔋 **Potential Harvestable Energy:** "
        f"**{harvestable_energy:.2f} Wh/m² (per hour)**"
    )

    st.caption(
        "Assumptions: 1-hour duration, 16% PV system efficiency, "
        "dew point estimated from temperature."
    )
