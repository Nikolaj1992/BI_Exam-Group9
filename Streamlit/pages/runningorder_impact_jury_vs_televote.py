import streamlit as st
import joblib
import pandas as pd

st.title("Eurovision Voting Analysis")
st.subheader("Impact of Performance Order on Jury and Televote Scores", divider='rainbow')

# Load pre-trained models (make sure to save them as .pkl files)
@st.cache_resource
def load_model(model_name):
    return joblib.load(f'../Models/{model_name}.pkl')  # Adjust the path to where your models are saved

jury_model = load_model("jury_running_order")
televote_model = load_model("televote_running_order")

def show_performance_order_impact():
    st.write("Enter the performance details below:")

    # Input fields for performance order
    year = st.number_input("Year", min_value=2000, max_value=2025, value=2023)
    final_draw_position = st.number_input("Final Draw Position", min_value=1, max_value=26, value=15)

    

    # Make prediction when button is pressed
    if st.button("Predict Jury and Televote Scores"):
        # Prepare input data as a DataFrame (excluding the target columns)
        input_data = pd.DataFrame([[year, final_draw_position]],
                                  columns=['year', 'final_draw_position'])
        
        # Predict using both the jury and televote models
        jury_prediction = jury_model.predict(input_data)
        televote_prediction = televote_model.predict(input_data)
        
        # Display results for Jury and Televote Predictions
        st.success(f"Predicted Jury Points: {jury_prediction[0]:.2f}")
        st.success(f"Predicted Televote Points: {televote_prediction[0]:.2f}")
        
        # Add a fun message based on the prediction (optional)
        if jury_prediction[0] > 10:
            st.info("The jury seems to favor your entry, nice job!")
        else:
            st.warning("Looks like the jury might not be as impressed. Better luck next year!")

        if televote_prediction[0] > 150:
            st.info("The audience is really loving your entry!")
        else:
            st.warning("Looks like the televote might not be in your favor. Try working on the performance!")

# Run the function to display the interface
show_performance_order_impact()
