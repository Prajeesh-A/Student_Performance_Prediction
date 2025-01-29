import pickle
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder

# Streamlit app
st.title("Score Prediction")

# Input fields for the user
name = st.text_input("Enter your name")
Hours_Studied = st.number_input("Hours Studied", min_value=0, max_value=100, value=4)
Attendance = st.number_input("Attendance", min_value=0, max_value=100, value=90)
Access_to_Resources_m = st.selectbox("Access to Resources", ['Low', 'Medium', 'High'])
Motivation_Level_m = st.selectbox("Motivation Level", ['Low', 'Medium', 'High'])

# Prepare the input data as a dictionary
input_data = {
    'Name': name,
    "Hours_Studied": Hours_Studied,
    "Attendance": Attendance,
    "Access_to_Resources": Access_to_Resources_m,
    "Motivation_Level": Motivation_Level_m
}

# Convert input data to DataFrame
new_data = pd.DataFrame([input_data])

lmh = {'Low': 0, 'Medium': 1, 'High': 2}

new_data['Access_to_Resources_m'] = new_data['Access_to_Resources'].map(lmh)
new_data['Motivation_Level_m'] = new_data['Motivation_Level'].map(lmh)

df =pd.read_csv('features.csv')
columns_list =  [col for col in df.columns if col not in 'unnamed: 0']

new_data = new_data.reindex(columns=columns_list, fill_value=0)

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

if st.button("Predict"):
    prediction = model.predict(new_data)

    st.markdown(f"""
        <div style='
            background-color: #f0f0f0; 
            padding: 10px; 
            border-radius: 5px; 
            text-align: center; 
            font-size: 20px; 
            color: blue;'>
            <strong>The predicted score of {name} = {prediction[0]}</strong>
        </div>
    """, unsafe_allow_html=True)