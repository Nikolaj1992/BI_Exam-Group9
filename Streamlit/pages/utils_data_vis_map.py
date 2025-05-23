import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static

# -------------------------
# App Title and Instructions
# -------------------------
st.set_page_config(page_title="Eurovision Ranking Map", layout="wide")
st.title("🎤 Eurovision Ranking Map")
st.markdown("Upload a CSV with `year`, `country`, and `final_place` columns to see how countries ranked on the map.")

# -------------------------
# Upload CSV
# -------------------------
uploaded_file = st.file_uploader("📁 Upload Eurovision CSV", type="csv")
if not uploaded_file:
    st.info("Please upload a file to continue.")
    st.stop()

df = pd.read_csv(uploaded_file)

# -------------------------
# Coordinates for Countries
# -------------------------
country_coords = {
    "Albania": [41.3275, 19.8189],
    "Andorra": [42.5078, 1.5211],
    "Armenia": [40.1792, 44.4991],
    "Australia": [-35.2809, 149.1300],
    "Austria": [48.2082, 16.3738],
    "Azerbaijan": [40.4093, 49.8671],
    "Belarus": [53.9006, 27.5590],
    "Belgium": [50.8503, 4.3517],
    "Bosnia and Herzegovina": [43.8563, 18.4131],
    "Bulgaria": [42.6977, 23.3219],
    "Croatia": [45.8150, 15.9819],
    "Cyprus": [35.1856, 33.3823],
    "Czech Republic": [50.0755, 14.4378],
    "Denmark": [55.6761, 12.5683],
    "Estonia": [59.4370, 24.7536],
    "Finland": [60.1695, 24.9354],
    "France": [48.8566, 2.3522],
    "Georgia": [41.7151, 44.8271],
    "Germany": [52.5200, 13.4050],
    "Greece": [37.9838, 23.7275],
    "Hungary": [47.4979, 19.0402],
    "Iceland": [64.1355, -21.8954],
    "Ireland": [53.3498, -6.2603],
    "Israel": [31.7683, 35.2137],
    "Italy": [41.9028, 12.4964],
    "Latvia": [56.9496, 24.1052],
    "Lithuania": [54.6872, 25.2797],
    "Luxembourg": [49.6117, 6.1319],
    "Malta": [35.8997, 14.5146],
    "Moldova": [47.0105, 28.8638],
    "Monaco": [43.7384, 7.4246],
    "Montenegro": [42.4304, 19.2594],
    "Netherlands": [52.3676, 4.9041],
    "North Macedonia": [41.9981, 21.4254],
    "Norway": [59.9139, 10.7522],
    "Poland": [52.2297, 21.0122],
    "Portugal": [38.7169, -9.1399],
    "Romania": [44.4268, 26.1025],
    "Russia": [55.7558, 37.6173],
    "San Marino": [43.9333, 12.4500],
    "Serbia": [44.7866, 20.4489],
    "Slovakia": [48.1486, 17.1077],
    "Slovenia": [46.0569, 14.5058],
    "Spain": [40.4168, -3.7038],
    "Sweden": [59.3293, 18.0686],
    "Switzerland": [46.9481, 7.4474],
    "Turkey": [39.9208, 32.8541],
    "Ukraine": [50.4501, 30.5234],
    "United Kingdom": [51.5074, -0.1278],
    "Yugoslavia": [44.7866, 20.4489]
}

# Define a color for each style
style_colors = {
    "pop": "pink",
    "ballad": "purple",
    "traditional": "green",
    "rock": "red",
    "dance": "orange"
}
# -------------------------
# Sidebar Filters
# -------------------------
st.sidebar.header("🧭 Filters")
year = st.sidebar.selectbox("Select Year", sorted(df["year"].unique(), reverse=True))
max_rank = st.sidebar.slider("Show countries ranked up to:", 1, 26, 10)
use_style_colors = st.sidebar.checkbox("🎨 Use music style colors", value=True)
# -------------------------
# Filtered Data
# -------------------------
filtered_df = df[(df["year"] == year) & (df["final_place"] <= max_rank)]

if filtered_df.empty:
    st.warning("No data for selected year and rank range.")
    st.stop()

# -------------------------
# Build Map
# -------------------------
m = folium.Map(location=[54, 15], zoom_start=4)

# Color function based on rank
def get_color(rank):
    if rank == 1:
        return "green"
    elif rank <= 3:
        return "orange"
    else:
        return "blue"

for _, row in filtered_df.iterrows():
    country = row["country"]
    rank = row["final_place"]
    coords = country_coords.get(country)

    if not coords:
        st.warning(f"Coordinates not found for {country}")
        continue

    if use_style_colors:
        # Style-based color
        music_style = str(row.get("style", "")).lower()
        icon_color = style_colors.get(music_style, "blue")
        popup = f"<b>{country}</b><br>Rank: {rank}<br>Style: {music_style.title()}"
    else:
        # Rank-based color
        if rank == 1:
            icon_color = "green"
        elif rank <= 3:
            icon_color = "orange"
        else:
            icon_color = "blue"
        popup = f"<b>{country}</b><br>Rank: {rank}"

    folium.Marker(
        location=coords,
        popup=popup,
        tooltip=country,
        icon=folium.Icon(color=icon_color)
    ).add_to(m)

    
# -------------------------
# Show Map and Data
# -------------------------
st.subheader(f"🌍 Eurovision {year} - Top {max_rank}")
folium_static(m)


# -------------------------
# Table Row Styling Function
# -------------------------
def highlight_style(row):
    color_map = {
        "pop": "mistyrose",
        "ballad": "lavender",
        "traditional": "honeydew",
        "rock": "lightcoral",
        "dance": "peachpuff"
    }
    style = str(row.get("style", "")).lower()
    color = color_map.get(style, "white")
    return [f"background-color: {color}"] * len(row)

with st.expander("📊 View Data Table"):
    table_data = filtered_df.sort_values("final_place").reset_index(drop=True)
    
    if use_style_colors:
        st.dataframe(table_data.style.apply(highlight_style, axis=1), use_container_width=True)
    else:
        st.dataframe(table_data, use_container_width=True)
