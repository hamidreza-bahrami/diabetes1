import streamlit as st
import pandas as pd
import numpy as np
import pickle 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import time
st.set_page_config(page_title='تشخیص دیابت - RoboAi', layout='centered', page_icon='🤖')

def load_model():
    with open('saved.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

model = data['model']
x = data['x']

def show_page():
    st.write("<h2 style='text-align: center; color: blue;'>مدل تشخیص دیابت در بانوان 📋</h2>", unsafe_allow_html=True)
    st.write("<h5 style='text-align: center; color: gray;'>Robo-Ai.ir طراحی شده توسط</h5>", unsafe_allow_html=True)
    st.link_button("Robo-Ai بازگشت به", "https://robo-ai.ir")
    container = st.container(border=True)
    container.write("<h6 style='text-align: right; color: gray;'>در بانوان II پرسشنامه تشخیص دیابت نوع 🩸</h6>", unsafe_allow_html=True)
    with st.sidebar:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(' ')
        with col2:
            st.image('img.png')
        with col3:
            st.write(' ')
        st.divider()
        st.write("<h4 style='text-align: center; color: blcak;'>تشخیص دیابت با هوش مصنوعی 🩸</h>", unsafe_allow_html=True)
        st.write("<h4 style='text-align: right; color: gray;'>ساخته شده با اطلاعات 420 کیس مبتلا به دیابت و سالم</h>", unsafe_allow_html=True)
        st.divider()
        st.write("<h5 style='text-align: center; color: black;'>طراحی و توسعه</h5>", unsafe_allow_html=True)
        st.write("<h5 style='text-align: center; color: black;'>حمیدرضا بهرامی</h5>", unsafe_allow_html=True)

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
        with st.chat_message("assistant"):
                with st.spinner('''درحال بررسی، لطفا صبور باشید'''):
                    time.sleep(2)
                    st.success(u'\u2713''بررسی انجام شد')
                    x = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])

        y_prediction = model.predict(x)
        if y_prediction == 1:
            text1 = '###بر اساس تحلیل من ، شما به دیابت نوع 2 مبتلا هستید'
            text2 = 'برای درمان به پزشک مراجعه کنید'
            text3 = 'Based On My Analysis, You Are Diagnosed With Type 2 Diabetes'
            text4 = 'Please Visit A Doctor As Soon As Possible'
            def stream_data1():
                for word in text1.split(" "):
                    yield word + " "
                    time.sleep(0.09)
            st.write_stream(stream_data1)
            def stream_data2():
                for word in text2.split(" "):
                    yield word + " "
                    time.sleep(0.09)
            st.write_stream(stream_data2)
            def stream_data3():
                for word in text3.split(" "):
                    yield word + " "
                    time.sleep(0.09)
            st.write_stream(stream_data3)
            def stream_data4():
                for word in text4.split(" "):
                    yield word + " "
                    time.sleep(0.09)
            st.write_stream(stream_data4)
            
        elif y_prediction == 0:
            text1 = 'بر اساس تحلیل من ، شما در سلامتی کامل هستید'
            text2 = 'Based On My Analysis, You are Totally Healthy'
            def stream_data1():
                for word in text1.split(" "):
                    yield word + " "
                    time.sleep(0.09)
            st.write_stream(stream_data1)
            def stream_data2():
                for word in text2.split(" "):
                    yield word + " "
                    time.sleep(0.09)
            st.write_stream(stream_data2)
show_page()
