import streamlit as st
import joblib
import pandas as pd

st.title("Linear Regression for Semifinals")
st.subheader("Eurovision Semifinal Place Predictor", divider='rainbow')

@st.cache_resource
def load_model():
    return joblib.load('../Models/semifinalpredictions.pkl') 

model = load_model()

def show_linear_regression():
    st.write("Enter the performance details below:")

    # Input fields for essential features
    year = st.number_input("Year", min_value=2000, max_value=2025, value=2023)
    semi_final = st.number_input("Semifinal", min_value=1, max_value=2, value=1)
    semi_draw_position = st.number_input("Semifinal Draw Position", min_value=1, max_value=26, value=15)
    semi_total_points = st.number_input("Total Points", min_value=0.0, value=250.0)

    # Input fields for optional features
    semi_televote_points = st.number_input("Televote Points", min_value=0.0, value=150.0)
    semi_jury_points = st.number_input("Jury Points", min_value=0.0, value=100.0)

    # Make prediction when button is pressed
    if st.button("Predict Final Place"):
        # Prepare input data as a DataFrame
        input_data = pd.DataFrame([[year, semi_final, semi_draw_position, semi_televote_points, semi_jury_points, semi_total_points]],
                                  columns=['year','semi_final', 'semi_draw_position', 'semi_televote_points', 
                                           'semi_jury_points', 'semi_total_points'])

        # Make prediction using the model
        prediction = model.predict(input_data)
        st.success(f"Predicted Semifinal Result: {prediction[0]:.2f}")
       


# Check if the prediction is in the top 10 - as in the entry would qualify to the grand final
        if prediction[0] < 1.1:
            st.success("You have won the semifinal - can you keep up the momentum?")
        
        if prediction[0] < 10:
            st.success("You have qualified for the grand final!")
        else:
            st.info("You have not qualified - maybe try changing up your selection method next year?")


# Run the function to display the interface
show_linear_regression()
