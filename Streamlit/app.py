import streamlit as st
from streamlit_option_menu import option_menu

from PIL import Image
logo = Image.open('./media/esc.jpg')

st.set_page_config(
    page_title="MP3",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Exam project")
st.subheader("This project is a analysis of the Eurovision Song Contest", divider="rainbow")
st.image(logo, width = 700 )

st.write("Made by: Jenny, David, Nikolaj and Patrick")
st.write("Sem 4, BI, 2025")


