import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.neighbors import NearestNeighbors

#SET PAGE NAME
st.set_page_config(page_title="Tourism Attraction Recommender", layout="centered")

#STYLING THE APPLICATION WITH CSS
st.markdown(
    """
    <style>
        body {
            background-color: #f0f2f6;
        }
        .main-container {
            background-color: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.15);
        }
        .stButton>button {
            background-color: #4B9EFF;
            color: white;
            font-size: 18px;
            padding: 10px 20px;
            border-radius: 8px;
        }
        .stButton>button:hover {
            background-color: #e43f3f;
        }
        .recommendation-box {
            background-color: #f8f9fa;
            padding: 10px;
            margin-top: 10px;
            border-radius: 8px;
            font-size: 16px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

#SIDE BAR INFO ABOUT THE APPLICATION
st.sidebar.header("ℹ️ About the App")
st.sidebar.write("""
🔹 **Welcome to the Tourism Attraction Recommender!**  
This system suggests **tourist attractions** based on user preferences.

### **🛠 How it Works?**  
1️⃣ Select a **User ID** from the dropdown.  
2️⃣ Click **"🔮 Get Recommendations"**.  
3️⃣ View the top **5 recommended attractions**.  

### **📊 Methodology:**  
✔ Uses **Collaborative Filtering**  
✔ Implements **KNN (K-Nearest Neighbors)**  
✔ Utilizes **Truncated SVD** for dimensionality reduction  

🚀 **Enjoy your personalized recommendations!**
""")


#LOAD DATA AND TRAINED MODEL
svd = joblib.load(r"D:\TRANSACTION PROJECT\svd1 for recommend.pkl")
knn_model = joblib.load(r"D:\TRANSACTION PROJECT\knn_model1 for recommend.pkl")
user_attraction_matrix = joblib.load(r"D:\TRANSACTION PROJECT\user_attraction_matrix1 for recommend.pkl")
user_attraction_matrix_reduced = joblib.load(r"D:\TRANSACTION PROJECT\user_attraction_matrix_reduced for recommend.pkl")

#STYLE THE TITLE OF APPLICATION
st.markdown(
    """
    <div style="
        text-align: center;
        background-color: #4B9EFF;
        color: white;
        padding: 15px;
        border-radius: 12px;
        font-size: 24px;
        font-weight: bold;
        box-shadow: 2px 4px 10px rgba(0,0,0,0.2);
    ">
        🏝️ TOURISM ATTRACTION RECOMMENDATION
    </div>
    """,
    unsafe_allow_html=True
)


# CREATE SELECT BOX FOR INPUT
st.markdown("<h3>Select User ID for Recommendations</h3>", unsafe_allow_html=True)
user_ids = user_attraction_matrix.index.tolist()
user_id = st.selectbox("Choose a User ID", user_ids)

#RECOMMENDATION FUNCTION 
def recommend_attractions(user_id, num_recommendations=5):
    user_idx = user_attraction_matrix.index.get_loc(user_id)
    
    distances, indices = knn_model.kneighbors(svd.transform(user_attraction_matrix.iloc[[user_idx]]), n_neighbors=5)
    
    similar_users = user_attraction_matrix.index[indices.flatten()[1:]]  # Exclude self
    user_ratings = user_attraction_matrix.loc[user_id]
    unseen_attractions = user_ratings[user_ratings == 0].index

    attraction_scores = {}
    for sim_user in similar_users:
        for attraction in unseen_attractions:
            attraction_scores[attraction] = attraction_scores.get(attraction, 0) + user_attraction_matrix.loc[sim_user, attraction]

    return sorted(attraction_scores, key=attraction_scores.get, reverse=True)[:num_recommendations]

#PREDICT THE RECOMMENDATION 
if st.button("🔮 Get Recommendations"):
    recommended = recommend_attractions(user_id)
    st.markdown("<h3> Recommended Attractions:</h3>", unsafe_allow_html=True)
    
    for i, attraction in enumerate(recommended, 1):
        st.markdown(f"<div class='recommendation-box'> ✨ {attraction}</div>", unsafe_allow_html=True)

#CREATE PREVIEW TABLE FOR USER FRIENDLY
st.subheader("📊 Preview of Tourism Data")
st.write("Below is a preview of the full tourism dataset used for predictions:")
st.dataframe(pd.read_csv(r"D:\TRANSACTION PROJECT\Full Tourism Data.csv"))