import streamlit as st
import joblib
import numpy as np

st.title("Classification")
st.subheader("Naives Bayes Eurovision Top 10 predictor", divider='rainbow')

def load_model():
    return joblib.load('../Models/bayes.pkl') 

model = load_model()

def show_classification ():
   # Input fields
    
    draw = st.number_input("Final Draw Position", min_value=1, max_value=27, value=25)
    televote_points = st.number_input("Televote Points", min_value=0, max_value=500, value=120)
    jury_points = st.number_input("Jury Points", min_value=0, max_value=500, value=130)


    # Prediction button
    if st.button("Predict Top 10 Placement"):
        sample = np.array([[draw, televote_points, jury_points]])
        prediction = model.predict(sample)
        result = "✅ YES – Likely Top 10!" if prediction[0] == 1 else "❌ NO – Not in Top 10"
        st.subheader("Prediction Result:")
        st.success(result)

show_classification()

