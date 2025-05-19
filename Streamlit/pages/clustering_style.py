import streamlit as st
import pandas as pd
import plotly.express as px
import pycountry

st.set_page_config(layout="wide")
st.title("Eurovision Clusters by Country")
st.subheader("Made out of Year, Country, Style, and final_total_points", divider='rainbow')

def load_data():
    return pd.read_csv('../Data/finalists_clustered.csv')

df = load_data()

# ——— Year filter ———
min_year = int(df['year'].min())
max_year = int(df['year'].max())
start, end = st.slider(
    "Select Year Range", 
    min_year, max_year, 
    (min_year, max_year)
)
df_filtered = df[(df['year'] >= start) & (df['year'] <= end)]
st.write(f"Showing data from {start} to {end} ({len(df_filtered)} entries)")

# ——— Aggregate to dominant cluster per country ———
country_cluster = (
    df_filtered
    .groupby('country')['Cluster']
    .agg(lambda x: x.mode()[0])
    .reset_index()
)

# ——— Map country names → ISO‑3 codes ———
def to_iso3(name):
    try:
        return pycountry.countries.lookup(name).alpha_3
    except:
        return None

country_cluster['iso_alpha'] = country_cluster['country'].apply(to_iso3)
country_cluster = country_cluster.dropna(subset=['iso_alpha'])

# ——— Plot choropleth ———
fig = px.choropleth(
    country_cluster,
    locations='iso_alpha',
    color='Cluster',
    hover_name='country',
    title=f'Eurovision Clusters by Country ({start}–{end})',
    color_discrete_sequence=px.colors.qualitative.Plotly
)

st.plotly_chart(fig, use_container_width=True)
