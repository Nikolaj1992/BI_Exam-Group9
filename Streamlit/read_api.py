import requests
import json
import streamlit as st
import os

def readAPI(url, params, headers, filepath):
    # Check if file already exists
    if os.path.exists(filepath):
        st.info(f"Loading cached data from: {filepath}")
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data

    # Make API request
    response = requests.get(url, params=params, headers=headers)
    if response.status_code == 200:
        data = response.json()

        # Save data to file
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)

        st.success(f"Data saved to {filepath}")
        return data
    else:
        st.error(f"API request failed with status code {response.status_code}")
        return None

def fetch_eurovision_data():
    st.title("Eurovision API Data Analyzer")

    # Year input
    year = st.number_input("Enter a Eurovision year (e.g. 2024):", min_value=1956, max_value=2100, value=2024, step=1)

    # Only fetch data when button is clicked
    if st.button("Fetch Eurovision Data"):
        # Ensure the Json directory exists
        json_dir = "Json"
        os.makedirs(json_dir, exist_ok=True)

        # Build file path
        filename = f"eurovision_{year}.json"
        filepath = os.path.join(json_dir, filename)

        # Construct URL
        url = f"https://eurovisionapi.runasp.net/api/contests/{year}"
        params = {}
        headers = {}

        # Read from API or cache
        data = readAPI(url, params, headers, filepath)

        # Display data
        if data:
            st.subheader(f"Data for Eurovision {year}")
            st.json(data)

            # Optional simplified display
            if isinstance(data, list):
                for item in data:
                    st.write(f"Year: {item.get('year')}")
                    st.write(f"Host: {item.get('hostCity')}")
                    st.write(f"Date: {item.get('date')}")
                    st.write("---")
            elif isinstance(data, dict):
                st.write("Available fields:")
                for key, value in data.items():
                    st.write(f"**{key}**: {value}")
