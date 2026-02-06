import streamlit as st
import joblib 

st.title("Water Potability Classifier")

model = joblib.load('model.joblib')

ph = st.number_input('pH', min_value=0.0, max_value=14.0, value=7.0)
hardness = st.number_input('Hardness', min_value=0.0, value=200.0)
solids = st.number_input('Solids', min_value=0.0, value=20000.0)
chloramines = st.number_input('Chloramines', min_value=0.0, value=7.0)
sulfate = st.number_input('Sulfate', min_value=0.0, value=300.0)
conductivity = st.number_input('Conductivity', min_value=0.0, value=400.0)
organic_carbon = st.number_input('Organic Carbon', min_value=0.0, value=15.0)
trihalomethanes = st.number_input('Trihalomethanes', min_value=0.0, value=60.0)
turbidity = st.number_input('Turbidity', min_value=0.0, value=4.0)

if st.button('Predict'):
    pred = model.predict([[ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity]])
    st.success(f'The predicted Potability is: {pred[0]}')
