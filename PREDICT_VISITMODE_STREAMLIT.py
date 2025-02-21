import streamlit as st
import pandas as pd
import joblib

#LOAD TRAINED MODEL AND ENCODER
model = joblib.load(r"D:\TRANSACTION PROJECT\BEST MODEL VISITMODE.pkl")
ohe = joblib.load(r"D:\TRANSACTION PROJECT\ohe_for_visitmode.pkl")
label_encoder = joblib.load(r"D:\TRANSACTION PROJECT\label_encoding_for_visitmode.pkl")
target_enc = joblib.load(r"D:\TRANSACTION PROJECT\target_encode_for_visitmode.pkl")

#DEFINE THE FEATURE DATA COLUMN
selected_features = ["UserId", "VisitYear", "VisitMonth", "VisitMode", "AttractionId", 
                     "ContenentId", "RegionId", "Attraction", "AttractionType", "AttractionTypeId"]

#STYLING THE APPLICATION WITH CSS
st.markdown(
    """
    <style>
        body {
            background-color: #f4f4f4;
            font-family: 'Arial', sans-serif;
        }
        .stApp {
            background: url("https://source.unsplash.com/1600x900/?travel,nature") no-repeat center fixed;
            background-size: cover;
        }
        .title {
            color: #ffffff;
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            background: rgba(0, 0, 0, 0.6);
            padding: 10px;
            border-radius: 10px;
        }
        /* Removes extra gray box from Streamlit columns */
        [data-testid="stVerticalBlock"] {
            background: transparent !important;
            box-shadow: none !important;
            padding: 0px !important;
        }
        .stButton>button {
            background: linear-gradient(to right, #ff416c, #ff4b2b);
            color: white;
            border-radius: 10px;
            font-size: 18px;
            font-weight: bold;
            padding: 10px 20px;
        }
        .stButton>button:hover {
            background: linear-gradient(to right, #ff4b2b, #ff416c);
        }
    </style>
    """,
    unsafe_allow_html=True
)

#SIDE BAR INFO ABOUT THE APPLICATION
st.sidebar.title("📌 About the App")
st.sidebar.info(
    """
    🏝️ **Visit Mode Prediction App**
    
    This application helps predict the **visit mode** of a user based on various inputs such as:
    
    - User ID
    - Visit Year & Month
    - Attraction & Region Details
    - Attraction Type & ID
    
    **How to Use?**  
    1️⃣ Fill in the required details  
    2️⃣ Click on "Predict Visit Mode"  
    3️⃣ Get the prediction result instantly ✅  

    🚀 Built using **Machine Learning & Streamlit**
    """
)
st.markdown('<h1 class="title">🏝️ Visit Mode Prediction App</h1>', unsafe_allow_html=True)
st.write("Enter details to predict the visit mode (Business, Family, Couples, Friends, etc.)")

#CREATE THE NUMERIC AND CATEGORICAL INPUT BOX
col1, col2 = st.columns(2)

with col1:
    user_id = st.number_input("🔹 User ID", min_value=1, step=1, format="%d")
    visit_year = st.number_input("📅 Visit Year", min_value=2000, max_value=2100, step=1, format="%d")
    visit_month = st.number_input("📆 Visit Month", min_value=1, max_value=12, step=1, format="%d")
    visit_mode = st.number_input("🚶‍♂️ Visit Mode (Encoded)", min_value=1, step=1, format="%d")
    attraction_id = st.number_input("📍 Attraction ID", min_value=1, step=1, format="%d")

with col2:
    continent_id = st.number_input("🌍 Continent ID", min_value=1, step=1, format="%d")
    region_id = st.number_input("🏙️ Region ID", min_value=1, step=1, format="%d")
    attraction = st.text_input("🎡 Attraction Name")
    attraction_type = st.text_input("🎭 Attraction Type")
    attraction_type_id = st.number_input("🆔 Attraction Type ID", min_value=1, step=1, format="%d")

# CREATE THE PREDICT BUTTON AND PREDICT THE TARGET WITH HELP OF ENCODER
if st.button("🔍 Predict Visit Mode"):
    if not all([user_id, visit_year, visit_month, visit_mode, attraction_id, 
                continent_id, region_id, attraction, attraction_type, attraction_type_id]):
        st.error("⚠️ Please enter all the inputs before predicting!")
    else:
        input_data = pd.DataFrame([[user_id, visit_year, visit_month, visit_mode, attraction_id, 
                                     continent_id, region_id, attraction, attraction_type, attraction_type_id]],
                                  columns=selected_features)

        categorical_features = ["VisitMode", "AttractionType"]
        encoded_features = ohe.transform(input_data[categorical_features])
        encoded_df = pd.DataFrame(encoded_features, columns=ohe.get_feature_names_out(categorical_features))

        input_data["Attraction"] = target_enc.transform(input_data["Attraction"])

        input_data = input_data.drop(columns=categorical_features)
        input_data = pd.concat([input_data.reset_index(drop=True), encoded_df], axis=1)

        prediction = model.predict(input_data)
        
        predicted_category = label_encoder.inverse_transform(prediction)[0]

        st.success(f"✅ Predicted Visit Mode: **{predicted_category}**")

#CREATE PREVIEW TABLE FOR USER FRIENDLY
st.subheader("📊 Preview of Tourism Data")
st.write("Below is a preview of the full tourism dataset used for predictions:")
st.dataframe(pd.read_csv(r"D:\TRANSACTION PROJECT\Full Tourism Data.csv"))
