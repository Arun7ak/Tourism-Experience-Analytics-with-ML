# Tourism Experience Analytics with Machine Learning

## Project Overview
This project involves combining multiple CSV files into a single dataset, integrating it with MySQL using XAMPP, cleaning the dataset, and training machine learning models for three key objectives:

1. **Regression:** Predicting attraction ratings.
2. **Classification:** User visit mode prediction.
3. **Recommendations:** Personalized attraction suggestions.

Additionally, a **Streamlit application** has been built to provide a user-friendly interface for interacting with the trained models.

---

## Objective 1: Regression - Predicting Attraction Ratings

### **Aim:**
Develop a regression model to predict the rating a user might give to a tourist attraction based on historical data, user demographics, and attraction features.

### **Use Cases:**
- Travel platforms can estimate user satisfaction for specific attractions.
- Identify attractions likely to receive lower ratings and take corrective actions.
- Personal travel guides can enhance user experience by suggesting attractions aligned with preferences.

### **Possible Inputs (Features):**
- **User demographics:** Continent, region, country, city.
- **Visit details:** Year, month, mode of visit (e.g., business, family, friends).
- **Attraction attributes:** Type (e.g., beaches, ruins), location, previous average ratings.

### **Target:**
- Predicted rating (on a scale, e.g., 1-5).

---

## Objective 2: Classification - User Visit Mode Prediction

### **Aim:**
Create a classification model to predict the mode of visit (e.g., business, family, couples, friends) based on user and attraction data.

### **Use Cases:**
- Travel platforms can personalize marketing campaigns based on user visit mode predictions.
- Hotels and attraction organizers can plan resources more efficiently.

### **Inputs (Features):**
- **User demographics:** Continent, region, country, city.
- **Attraction characteristics:** Type, popularity, previous visitor demographics.
- **Historical visit data:** Month, year, previous visit modes.

### **Target:**
- Visit mode categories (e.g., Business, Family, Couples, Friends, etc.).

---

## Objective 3: Recommendations - Personalized Attraction Suggestions

### **Objective:**
Develop a recommendation system to suggest tourist attractions based on a user's historical preferences and similar users’ preferences.

### **Use Cases:**
- Travel platforms can guide users toward attractions they are most likely to enjoy, increasing engagement.
- Destination management organizations can promote attractions aligned with user interests.

### **Types of Recommendation Approaches:**
- **Collaborative Filtering:** Recommend attractions based on similar users' ratings and preferences.
- **Content-Based Filtering:** Suggest attractions similar to those already visited by the user based on features like attraction type, location, and amenities.
- **Hybrid Systems:** Combine collaborative and content-based methods for improved accuracy.

### **Inputs (Features):**
- **User visit history:** Attractions visited, ratings given.
- **Attraction features:** Type, location, popularity.
- **Similar user data:** Travel patterns and preferences.

### **Output:**
- Ranked list of recommended attractions.

---

## Streamlit Application
A **Streamlit app** has been developed to provide a user-friendly interface for interacting with the machine learning models. Users can input details and receive predictions for:
- Attraction ratings.
- Visit modes.
- Recommended attractions.

---

## Technologies Used
- **Python** (Pandas, NumPy, Scikit-Learn, Joblib) for data processing and model training.
- **MySQL & XAMPP** for database integration.
- **Streamlit** for building the interactive web application.

---

## How to Run the Project
1. Clone the repository from GitHub.
2. Install required dependencies using `pip install -r requirements.txt`.
3. Start MySQL and XAMPP to enable database connectivity.
4. Run the Streamlit application using `streamlit run app.py`.

---

## Repository Structure
```
├── data/                 # Raw and cleaned dataset files
├── models/               # Trained machine learning models
├── notebooks/            # Jupyter notebooks for EDA and model training
├── src/                  # Source code for data processing and ML models
├── app.py                # Streamlit application
├── requirements.txt      # Dependencies
└── README.md             # Project documentation
```

