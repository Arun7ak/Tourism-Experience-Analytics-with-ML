import streamlit as st
import joblib
import pandas as pd

# LOAD THE TRAINED MODEL
model_path = r"D:\TRANSACTION PROJECT\rating rf model.pkl"
rf_model = joblib.load(model_path)

#CREATE TITLE
st.title("Rating Prediction App")
st.write("Enter the details below to predict the rating.")

#CREATE INPUT BOX
VisitYear = st.number_input("Visit Year", min_value=2000, max_value=2100, step=1)
VisitMonth = st.number_input("Visit Month", min_value=1, max_value=12, step=1)
VisitModeName = st.text_input("Visit Mode Name")
AttractionId = st.number_input("Attraction ID", min_value=1, step=1)
AttractionType = st.text_input("Attraction Type")
Country = st.text_input("Country")
CityName = st.text_input("City Name")

#CREATE PREDICT BUTTON
if st.button("Predict Rating"):
    # Create DataFrame for model input
    input_data = pd.DataFrame({
        "VisitYear": [VisitYear],
        "VisitMonth": [VisitMonth],
        "VisitModeName": [VisitModeName],
        "AttractionId": [AttractionId],
        "AttractionType": [AttractionType],
        "Country": [Country],
        "CityName": [CityName]
    })
    
    #MAKE PREDICTION
    prediction = rf_model.predict(input_data)
    
    #DISPLAY THE PREDICTION
    st.write(f"Predicted Rating: {prediction[0]}")
