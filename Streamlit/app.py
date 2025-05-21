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

st.title("MP3")
st.subheader("Linear Regression, Classification, Clustering", divider="rainbow")
st.image(logo, width = 700 )

st.subheader("This project is a analysis of the Eurovision Song Contest", divider="rainbow")
st.write("Made by: Jenny, David, Nikolaj and Patrick")
st.write("Sem 4, BI, 2025")

st.markdown("""
### Key Analyses:
- Linear Regression for Eurovision Final Results
- Classification of Eurovision Winners
- Clustering of Countries by Voting Patterns
- **Performance Order Analysis** (New!) - Explore how the order of performances influences final results!

Navigate through the analyses using the sidebar.
""")


