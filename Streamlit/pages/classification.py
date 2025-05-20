import streamlit as st
import joblib
import pandas as pd
from PIL import Image
import os

image_path = '../Images/NB_confusion_matrix.png'

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
        sample = [[draw, televote_points, jury_points]]
        sample_df = pd.DataFrame(sample, columns=['final_draw_position', 'final_televote_points', 'final_jury_points'])
        prediction = model.predict(sample_df)
        result = "✅ YES – Likely Top 10!" if prediction[0] == 1 else "❌ NO – Not in Top 10"
        st.subheader("Prediction Result:")
        st.success(result)

show_classification()


# Confussion Matrix
def load_confusion_matrix_image():
    image_path = '../Images/NB_confusion_matrix.png'
    if os.path.exists(image_path):
        return Image.open(image_path)
    else:
        return None

if st.button("Show Confusion Matrix and Metrics"):
    img = load_confusion_matrix_image()
    if img:
        st.image(img, caption='Naive Bayes Confusion Matrix', width=600)
    else:
        st.error("Confusion matrix image not found.")
    
# Classification report 
classification_report_text = """
Accuracy: 0.9028
              precision    recall  f1-score   support

           0       0.88      0.95      0.91        37
           1       0.94      0.86      0.90        35

    accuracy                           0.90        72
   macro avg       0.91      0.90      0.90        72
weighted avg       0.91      0.90      0.90        72
"""

st.subheader("Classification Report")
st.code(classification_report_text)