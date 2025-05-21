import streamlit as st
import sys
import os
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_core.vectorstores import InMemoryVectorStore


# Add the utils folder (two levels up) to sys.path
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'modules', 'utils'))
sys.path.append(utils_path)
from helpers import *

st.set_page_config(page_title="Web Q&A with LLM", layout="wide")
st.title("🌐 Web Reader")
st.subheader("So what does the internet has to say? -lets find out", divider='rainbow')

default_url = 'https://escinsight.com/2023/04/06/deeper-look-eurovisions-running-order/'

st.markdown("### 🔍 Select Content Source")
url_to_use = default_url

with st.expander("🔧 Advanced: Use a custom URL"):
    user_url = st.text_input("Custom URL:", placeholder="https://example.com")
    if user_url.strip():
        url_to_use = user_url.strip()

st.caption(f"🔗 Using: {url_to_use}")

question = st.text_input("Ask a question based on the webpage:")

if question:
    with st.spinner("Loading, processing, and answering..."):
        try:
            doc = load_web_page(url_to_use)
            text_chunks = split_web_text(doc)
            store_web_docs(text_chunks)

            retrieved = retrieve_docs(question)
            if retrieved:
                answer = answer_question(question, retrieved)
                st.markdown("### 🤖 Answer")
                st.write(answer)
                with st.expander("🔍 Retrieved context chunks"):
                    for doc in retrieved:
                        st.markdown(doc.page_content)
            else:
                st.warning("No relevant documents were found.")
        except Exception as e:
            st.error(f"Error processing: {e}")
