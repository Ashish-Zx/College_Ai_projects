import streamlit as st
import joblib 

st.title("Lab 1 : Iris Flower Classifier")

model =joblib.load('model.joblib')
sp_length = st.number_input('Sepal Length', min_value=0.0, max_value=10.0, value=5.0)
sp_width = st.number_input('Sepal Width', min_value=0.0, max_value=10.0, value=3.5)
pt_length = st.number_input('Petal Length', min_value=0.0, max_value=10.0, value=1.4)
pt_width = st.number_input('Petal Width', min_value=0.0, max_value=10.0, value=0.2)

if st.button('Predict'):
    pred = model.predict([[sp_length, sp_width, pt_length, pt_width]])
    st.success(f'The predicted species is: {pred[0]}')