# Save this as 'Streamlit/pages/decision_tree_app.py'

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Page config
st.set_page_config(page_title="Eurovision Top 10 Predictor", layout="wide")

# Function to load the model
@st.cache_resource
def load_model():
    # Get the absolute path to the model
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Models/gradient_boosting_eurovision.pkl'))
    return joblib.load(model_path)

# Load data
@st.cache_data
def load_data():
    # Get the absolute path to the data
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Data/finalists_cleaned.csv'))
    return pd.read_csv(data_path)

# Title
st.title("Eurovision Top 10 Predictor")
st.write("Predict if a country will finish in the Top 10 at Eurovision Song Contest")

# Load model and data
try:
    model = load_model()
    df = load_data()
    
    # Sidebar for inputs
    st.sidebar.header("Make a Prediction")
    
    # Get unique values for dropdowns
    countries = sorted(df['country'].unique())
    styles = sorted(df['style'].unique())
    years = sorted(df['year'].unique())
    years = list(range(min(years), max(years) + 5))  # Add some future years
    
    # Input fields
    selected_country = st.sidebar.selectbox("Select Country", countries)
    selected_style = st.sidebar.selectbox("Select Performance Style", styles)
    selected_year = st.sidebar.selectbox("Select Year", years)
    draw_position = st.sidebar.slider("Draw Position", 1, 26, 10)
    
    # Create sample for prediction
    sample = pd.DataFrame({
        'style': [selected_style],
        'year': [selected_year],
        'country': [selected_country],
        'final_draw_position': [draw_position]
    })
    
    # Predict button
    if st.sidebar.button("Predict Top 10 Finish"):
        prediction = model.predict(sample)[0]
        prediction_proba = model.predict_proba(sample)[0]
        
        # Map the prediction to categories
        category_map = {
            0: "Not Top 10 (places 11-26)",
            1: "Top 10 (places 1-10)"
        }
        
        # Display prediction
        st.markdown("## Prediction Results")
        
        # Get the predicted category and probability
        predicted_category = category_map[prediction]
        probability = prediction_proba[prediction] * 100
        
        # Show prediction with emoji and probability
        if prediction == 1:
            st.success(f"### {selected_country} is predicted to finish in the **Top 10** 🏆")
        else:
            st.warning(f"### {selected_country} is predicted to **NOT** finish in the Top 10 😢")
            
        st.write(f"Confidence: {probability:.1f}%")
        
        # Show probabilities as a bar chart
        prob_df = pd.DataFrame({
            'Category': ["Not Top 10", "Top 10"],
            'Probability': [prediction_proba[0]*100, prediction_proba[1]*100]
        })
        
        fig, ax = plt.subplots(figsize=(10, 4))
        bar_colors = ['#ff9999', '#66b3ff'] if prediction == 1 else ['#66b3ff', '#ff9999']
        sns.barplot(x='Category', y='Probability', data=prob_df, palette=bar_colors, ax=ax)
        ax.set_ylabel('Probability (%)')
        ax.set_ylim(0, 100)
        for i, v in enumerate(prob_df['Probability']):
            ax.text(i, v + 2, f"{v:.1f}%", ha='center')
        st.pyplot(fig)
        
        # Show historical data for context
        st.markdown("### Historical Performance")
        country_history = df[df['country'] == selected_country].sort_values('year', ascending=False)
        
        if not country_history.empty:
            st.write(f"Recent performance of {selected_country} in Eurovision:")
            
            # Add a "Top 10" column for the historical data
            country_history['Top 10'] = country_history['final_place'] <= 10
            country_history['Top 10'] = country_history['Top 10'].map({True: 'Yes ✓', False: 'No ✗'})
            
            st.dataframe(country_history[['year', 'style', 'final_draw_position', 'final_place', 'Top 10']].head(5))
            
            # Plot historical placements
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.lineplot(x='year', y='final_place', data=country_history, marker='o', ax=ax)
            ax.axhline(y=10.5, color='r', linestyle='--', alpha=0.7, label='Top 10 Threshold')
            ax.invert_yaxis()  # Lower is better (1st place at top)
            plt.title(f"Historical Placements for {selected_country}")
            plt.ylabel("Final Place")
            plt.xlabel("Year")
            plt.grid(True)
            plt.legend()
            st.pyplot(fig)
    
    # Model information section
    st.markdown("## Model Information")
    
    tab1, tab2 = st.tabs(["About the Model", "Performance Metrics"])
    
    with tab1:
        st.write("This classifier predicts whether a country will finish in the Top 10 at Eurovision.")
        st.write("It uses features like the country, performance style, year, and draw position to make predictions.")
        st.write("The model was trained using a Gradient Boosting Classifier on historical Eurovision data.")
    
    with tab2:
        st.write("Model performance metrics on test data:")
        st.write("----------------------------")
        # Update these values with the actual metrics from your binary classifier
        st.write("Accuracy: 0.67")
        st.write("Precision: 0.65")
        st.write("Recall: 0.66")
        st.write("F1 Score: 0.65")
        
        # Add a note about the metrics
        st.info("These metrics indicate how well the model distinguishes between countries that finish in the Top 10 versus those that don't.")
except Exception as e:
    st.error(f"Error loading model or data: {e}")
    # Show model path for debugging
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Models/gradient_boosting_eurovision.pkl'))
    st.write(f"Looking for model at: {model_path}")
    st.write(f"Current directory: {os.getcwd()}")
    if os.path.exists(os.path.join(os.path.dirname(__file__), '../../Models')):
        st.write(f"Files in Models directory: {os.listdir(os.path.join(os.path.dirname(__file__), '../../Models'))}")
