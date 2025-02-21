import streamlit as st
import joblib
import pandas as pd
import numpy as np

#LOAD TRAINED MODEL AND ENCODER
model_path = r"D:\TRANSACTION PROJECT\BEST MODEL FOR TOURISM.pkl"
ohe_path = r"D:\TRANSACTION PROJECT\ONE HOT ENCODING RATING.pkl"
target_enc_path = r"D:\TRANSACTION PROJECT\TARGET ENCODING RATING.pkl"
scaler_path = r"D:\TRANSACTION PROJECT\STANDARD SCALAR RATING.pkl"

best_xgb = joblib.load(model_path)
ohe = joblib.load(ohe_path)
target_enc = joblib.load(target_enc_path)
scaler = joblib.load(scaler_path)

#STYLING THE APPLICATION WITH CSS
st.markdown("""
    <style>
      .block-container {
        padding-top: 0rem !important;
    }
    body {
        background-color: #f0f2f6;
    }
    .main {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 5px 5px 15px rgba(0,0,0,0.2);
    }
    .title {
        text-align: center;
        color: #1f77b4;
        font-size: 30px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        color: gray;
        font-size: 18px;
        margin-bottom: 30px;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        border-radius: 10px;
        font-size: 16px;
        padding: 8px 20px;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #135a91;
    }
    </style>
""", unsafe_allow_html=True)

# CREATE THE TITLE AND LAYOUT
st.markdown("<div class='main'>", unsafe_allow_html=True)

st.markdown("<p class='title'>🏝️ TOURISM RATING PREDICTION APP</p>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Enter details below to predict the tourism rating.</p>", unsafe_allow_html=True)
 
#CREATE THE INPUT BOX 
col1, col2 = st.columns(2)

with col1:
    VisitYear = st.number_input("📅 Visit Year", min_value=2000, max_value=2100, step=1, help="Select the year of visit")
    VisitMonth = st.number_input("📆 Visit Month", min_value=1, max_value=12, step=1, help="Select the month of visit")
    VisitModeName = st.text_input("🚗 Visit Mode Name", placeholder="e.g., Car, Train, Flight")
    AttractionId = st.number_input("🏞️ Attraction ID", min_value=1, step=1, help="Enter attraction ID")

with col2:
    Attraction = st.text_input("🎡 Attraction Name", placeholder="e.g., Eiffel Tower")
    AttractionType = st.text_input("🏛️ Attraction Type", placeholder="e.g., Historic Sites, Beaches")
    CountryId = st.number_input("🌍 Country ID", min_value=1, step=1, help="Enter the country ID")
    RegionId = st.number_input("📍 Region ID", min_value=1, step=1, help="Enter the region ID")

# ALLIGN THE INPUT BOX IN CENTRE
st.markdown("<p style='text-align: center;'>", unsafe_allow_html=True)

#CREATE THE PREDICT BUTTON AND PREDICT THE TARGET WITH HELP OF ENCODER
if st.button("🔍 Predict Rating"):
    if not all([VisitYear, VisitMonth, VisitModeName, AttractionId, Attraction, AttractionType, CountryId, RegionId]):
        st.error("⚠️ Please provide all inputs before predicting!")
    else:
        input_data = pd.DataFrame({
            "VisitYear": [VisitYear],
            "VisitMonth": [VisitMonth],
            "VisitModeName": [VisitModeName],
            "AttractionId": [AttractionId],
            "Attraction": [Attraction],
            "AttractionType": [AttractionType],
            "CountryId": [CountryId],
            "RegionId": [RegionId]
        })

        encoded_features = ohe.transform(input_data[["VisitModeName", "AttractionType"]])
        encoded_df = pd.DataFrame(encoded_features, columns=ohe.get_feature_names_out(["VisitModeName", "AttractionType"]))

        input_data["Attraction"] = target_enc.transform(input_data["Attraction"])

        input_data = input_data.drop(columns=["VisitModeName", "AttractionType"])
        
        input_data = pd.concat([input_data, encoded_df], axis=1)

        input_scaled = scaler.transform(input_data)

        prediction = best_xgb.predict(input_scaled)

        st.success(f"⭐ Predicted Tourism Rating: {prediction[0]:.2f}")

st.markdown("</p>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

#CREATE PREVIEW TABLE FOR USER FRIENDLY
st.subheader("📊 Preview of Tourism Data")
st.write("Below is a preview of the full tourism dataset used for predictions:")
st.dataframe(pd.read_csv(r"D:\TRANSACTION PROJECT\Full Tourism Data.csv"))
