import ollama
import streamlit as st
from ollama import chat
from ollama_ocr import OCRProcessor

llm = 'llama3.2-vision:11b'

# Function to analyze a participant label
def participant_analyzer():
    st.title("participant Analyzer")
    uploaded_file = st.file_uploader("Upload an image of a participant or band", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        with st.spinner("Analyzing the picture..."):
            result = explain(uploaded_file.read())
            st.subheader("Picture Analysis")
            st.write(result)

# Function to call the vision model
def explain(image_bytes):
    response = ollama.chat(
        model=llm, 
        messages=[{
            'role': 'user',
            'content': 'who won final in 2000',
            'images': [image_bytes]
        }]
    )
    return response['message']['content']  # Adapt if format differs
