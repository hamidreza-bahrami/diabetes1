import streamlit as st
import pandas as pd
import numpy as np
import pickle 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def load_model():
    with open('saved.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

model = data['model']
x = data['x']

def show_page():
    st.write("<h1 style='text-align: center; color: black;'>مدل تشخیص دیابت در بانوان</h1>", unsafe_allow_html=True)
    st.write("<h2 style='text-align: center; color: gray;'>علائم خود را وارد کنید</h2>", unsafe_allow_html=True)

    Pregnancies = st.slider('تعداد بارداری', 0.0, 10.0, 1.0)

    Glucose = st.slider('سطح گلوکز خون', 70.0, 250.0, 75.0)

    BloodPressure = st.slider('فشار خون', 40.0, 100.0, 45.0)

    SkinThickness = st.slider('ضخامت پوست', 0.0, 5.0, 1.0)

    Insulin = st.slider('سطح انسولین خون', 0.0, 200.0, 1.0)

    BMI = st.slider('شاخص توده بدنی', 15.0, 50.0, 18.0)

    DiabetesPedigreeFunction = st.slider('شاخص احتمال ابتلا بر اثر ژنتیک', 0.08, 2.50, 0.1)

    Age = st.slider('سن', 18.0, 70.0, 18.0)

    button = st.button('معاینه و تشخیص')
    if button:
        x = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])

        y_prediction = model.predict(x)
        if y_prediction == 1:
            st.write("<h4 style='text-align: center; color: gray;'>بر اساس داده های وارد شده، شما به دیابت مبتلا هستید</h4>", unsafe_allow_html=True)
            st.write("<h5 style='text-align: center; color: gray;'>برای درمان به پزشک مراجعه کنید</h5>", unsafe_allow_html=True)
        elif y_prediction == 0:
            st.write("<h4 style='text-align: center; color: gray;'>بر اساس داده های وارد شده، شما در سلامتی کامل هستید</h4>", unsafe_allow_html=True)

show_page()
