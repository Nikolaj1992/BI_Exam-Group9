import streamlit as st
from streamlit_option_menu import option_menu

from PIL import Image
logo = Image.open('./media/esc.jpg')

st.set_page_config(
    page_title="Exam",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Exam project")
st.subheader("This project is a analysis of the Eurovision Song Contest", divider="rainbow")
st.image(logo, width = 550 )

st.write("Made by: Jenny, David, Nikolaj and Patrick")
st.write("Sem 4, BI, 2025")

st.markdown("""
### Key Analyses:
- Linear Regression for Eurovision Final Results
- Classification of Eurovision Winners
- Clustering of Countries by Voting Patterns 
- Clustering for Style and Country Patterns
- Death Slot Analysis
- Performance Order Analysis
- Style impact on Final results

### Predictors 
- Classification Decision Tree
- Classification Naives Bayes
- Linear Regression 

### Optionals
- Image Reader
- Api reader connected to Eurovision API
- Web Reader
- Q&A PDF

Navigate through the analyses using the sidebar.
""")


