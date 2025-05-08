import streamlit as st
import read_image
import read_api


# Function to display the homepage content
def show_homepage():
    st.title('Homepage')
    st.write('This Project is an analysis of European Song contest.')
    
    st.write("Made by: Jenny, David, Nikolaj and Patrick")

# Main function that runs the app
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Homepage", "Picture Analyzer", "ApiReader"])

    if page == "Homepage":
        show_homepage()
    elif page == "Picture Analyzer":
        read_image.participant_analyzer()
    elif page =="ApiReader":
        read_api.fetch_eurovision_data()
    

if __name__ == "__main__":
    main()