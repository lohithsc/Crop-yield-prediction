import pandas as pd
import joblib
import streamlit as st

# =========================
# LOAD MODEL
# =========================
data = joblib.load("crop_yield_model1.pkl")

model = data["model"]
preprocessor = data["preprocessor"]

# =========================
# PREDICTION FUNCTION
# =========================
def predict_yield(input_df):

    # Reorder columns EXACTLY like training
    input_df = input_df[[
        'temperature',
        'rainfall',
        'area',
        'crop_year',
        'season',
        'crop',
        'state',
        'district'
    ]]

    # Transform input
    X_trans = preprocessor.transform(input_df)

    # Predict
    prediction = model.predict(X_trans)[0]

    return prediction

# =========================
# STREAMLIT UI
# =========================
st.title("🌾 Crop Yield Prediction")

state = st.text_input("State")
district = st.text_input("District")
crop = st.text_input("Crop")

crop_year = st.number_input(
    "Crop Year",
    min_value=1990,
    max_value=2030,
    value=2020
)

season = st.selectbox(
    "Season",
    ["Kharif", "Rabi", "Whole Year"]
)

temperature = st.number_input(
    "Temperature (°C)",
    value=25.0
)

rainfall = st.number_input(
    "Rainfall (mm)",
    value=1000.0
)

area = st.number_input(
    "Area",
    value=100.0
)

# =========================
# PREDICT BUTTON
# =========================
if st.button("Predict Yield"):

    try:

        # IMPORTANT:
        # Match EXACT values used during training
        season_map = {
            "Kharif": "Kharif",
            "Rabi": "Rabi",
            "Whole Year": "whole_year"
        }

        input_df = pd.DataFrame([{
            "temperature": float(temperature),
            "rainfall": float(rainfall),
            "area": float(area),
            "crop_year": int(crop_year),
            "season": season_map[season],
            "crop": crop.strip(),
            "state": state.strip().upper(),
            "district": district.strip().upper()
        }])

        st.write("Input Data")
        st.dataframe(input_df)

        prediction = predict_yield(input_df)

        st.success(f"🌾 Predicted Yield: {prediction:.2f}")

    except Exception as e:
        st.error(f"Error: {e}")