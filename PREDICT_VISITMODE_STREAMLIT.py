import streamlit as st
import joblib
import pandas as pd

#LOAD THE TRAINED MODEL
model_path = r"D:\TRANSACTION PROJECT\rating rf model.pkl"
rf_model = joblib.load(model_path)

#CREATE THE TITLE
st.title("Visit Mode Prediction App")
st.write("Enter the details below to predict the Visit Mode Name.")

#CREATE THE INPUT BOX
UserId = st.number_input("User ID", min_value=1, step=1)
VisitYear = st.number_input("Visit Year", min_value=2000, max_value=2100, step=1)
VisitMonth = st.number_input("Visit Month", min_value=1, max_value=12, step=1)
AttractionId = st.number_input("Attraction ID", min_value=1, step=1)
Contenent = st.text_input("Contenent")
Region = st.text_input("Region")
Country = st.text_input("Country")
CityName = st.text_input("City Name")
Attraction = st.text_input("Attraction")
AttractionType = st.text_input("Attraction Type")

#CREATE THE PREDICT BUTTON AND PREDICT THE OUTPUT
if st.button("Predict Visit Mode Name"):
    input_data = pd.DataFrame({
        "UserId": [UserId],
        "VisitYear": [VisitYear],
        "VisitMonth": [VisitMonth],
        "AttractionId": [AttractionId],
        "Contenent": [Contenent],
        "Region": [Region],
        "Country": [Country],
        "CityName": [CityName],
        "Attraction": [Attraction],
        "AttractionType": [AttractionType]
    })
    prediction = rf_model.predict(input_data)
    
    #DISPLAY THE PREDICTION
    st.write(f"Predicted Visit Mode Name: {prediction[0]}")
