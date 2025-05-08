import ollama
import streamlit as st
from ollama import chat
from ollama_ocr import OCRProcessor

llm = 'llama3.2-vision:11b'

# Function to analyze a participant label
def participant_analyzer():
    st.title("Participant Analyzer")
    
    uploaded_file = st.file_uploader("Upload an image of a participant or band", type=["jpg", "jpeg", "png"])

    # Add input field for the user to type their question
    user_prompt = st.text_input("Ask a question about the image")

    if uploaded_file is not None and user_prompt:
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        with st.spinner("Analyzing the picture..."):
            result = explain(uploaded_file.read(), user_prompt)
            st.subheader("Picture Analysis")
            st.write(result)

# Function to call the vision model
def explain(image_bytes, prompt):
    response = ollama.chat(
        model=llm,
        messages=[{
            'role': 'user',
            'content': prompt,
            'images': [image_bytes]
        }]
    )
    return response['message']['content'] # Adapt if format differs
