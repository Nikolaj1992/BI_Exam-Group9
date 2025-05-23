import streamlit as st
import os
import sys
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_core.vectorstores import InMemoryVectorStore



# Get the absolute path to the utils directory
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'modules', 'utils'))
sys.path.append(utils_path)
from helpers import *

# Template for the assistant
template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""

# Directories
data = 'data/'
media = 'Media/'

@st.cache_data
def process_pdf_for_embeddings(file_path):
    text = parse_pdf(file_path, media)
    chunked = split_pdf_text(text)
    return chunked

def main():
    st.title("📄 RAG Chat with PDF")

    embeddings = OllamaEmbeddings(model="llama3.2:3b")
    vecDB = InMemoryVectorStore(embeddings)
    llm = OllamaLLM(model="gemma3:12b")

    st.subheader("📎 Upload PDF for Question Answering", divider='rainbow')
    uploaded_file = st.file_uploader("Drop your PDF file here", type=["pdf"], accept_multiple_files=False)

    if uploaded_file:
        # Save uploaded file once
        file_path = os.path.join(data, uploaded_file.name)
        upload_pdf(uploaded_file)

        # Let user input question anytime after upload
        question = st.text_input("💬 Ask a question about the PDF...")

        if question:
            with st.status("⚙️ Processing PDF and generating answer..."):
                # Parse PDF & create embeddings once
                text = parse_pdf(file_path, media)
                chunked = split_pdf_text(text)
                store_pdf_docs(chunked, vecDB)

                # Retrieve and answer
                retrieved_docs = retrieve_docs(question, vecDB)
                answer = answer_question(question, retrieved_docs, llm)
                st.markdown(f"**Answer:** {answer}")

    else:
        st.info("📥 Please upload a PDF file to begin.")
        
if __name__ == "__main__":
    main()
