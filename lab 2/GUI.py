import streamlit as st
import joblib

model =joblib.load ('model.joblib')

st.title("NEWS CATEGORIES PREDICTION")
st.markdown("### Enter News below")
 input_text =st.text_area(
  label ="",max_chars =10000,height=300

 )
if st.button("Predict Category")
   prediction= model.predict(input_text)