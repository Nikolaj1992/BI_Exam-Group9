import streamlit as st
import joblib
import pandas as pd

st.title("Linear Regression")
st.subheader("Eurovision Final Place Predictor", divider='rainbow')

# Load trained model

def load_model():
    return joblib.load('../Models/finalistfit.pkl') 

model = load_model()

def show_linear_regression():
    st.write("Enter the performance details below:")

    # Input fields for essential features
    
    draw = st.number_input("Final Draw Position", min_value=1, max_value=27, value=15)
    televote_points = st.number_input("Televote Points", min_value=0, max_value=500, value=120)
    jury_points = st.number_input("Jury Points", min_value=0, max_value=500, value=130)
    televote_votes = st.number_input("Televote Votes", min_value=0, max_value=50, value=25)
    jury_votes = st.number_input("Jury Votes", min_value=0, max_value=50, value=25)

    # Make prediction when button is pressed
    if st.button("Predict Final Place"):
        # Prepare input data as a DataFrame
        input_data = pd.DataFrame([[ draw, televote_points, jury_points,
                                    televote_votes, jury_votes]],
                                  columns=[ 'final_draw_position', 'final_televote_points', 
                                           'final_jury_points', 'final_televote_votes', 
                                           'final_jury_votes'])

        # Make prediction using the model
        prediction = model.predict(input_data)[0]
        prediction = max(1,round(prediction)) #ensure its atleast 1 and a whole number
        st.success(f"Predicted Final Place: {prediction}")

# Run the function to display the interface
show_linear_regression()