import streamlit as st

import numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt, joblib
import plotly.express as px
import plotly.graph_objs as gph
import os

from sklearn.cluster import  MeanShift, estimate_bandwidth
from sklearn import metrics
from scipy.spatial.distance import cdist
from sklearn.preprocessing import LabelEncoder

st.title("Clustering - Mean-Shift")
st.subheader("See how different styles have performed over the years. Different styles in the same cluster may have had similarities in either performance or musical characteristics. Year selection allows us to get a general idea of how style trends have changed over time.", divider='rainbow')

# Load model and data
model = joblib.load('../Models/meanshift.pkl')
le = joblib.load('../Models/label_encoder.pkl')

df = pd.read_csv('../Data/finalists_cleaned.csv', encoding='windows-1252')
# df = pd.read_csv('Data/finalists_cleaned.csv', encoding='windows-1252')
df_clean = df[['style', 'final_place', 'year']].dropna()
df_clean['style_encoded'] = le.transform(df_clean['style'])
df_clean['style_decoded'] = df_clean['style']  # Add readable label for plotting

# Sidebar: year selection
years = sorted(df_clean['year'].unique())
selected_year = st.sidebar.selectbox("Select year", ["All Years"] + years)

# Filter
if selected_year != "All Years":
    df_filtered = df_clean[df_clean['year'] == selected_year]
else:
    df_filtered = df_clean

# Predict clusters
X = df_filtered[['style_encoded', 'final_place', 'year']].values
labels = model.predict(X)
df_filtered['cluster'] = labels

# Plot
# fig = px.scatter(
#    df_filtered, x='style_decoded', y='final_place', 
#    color='cluster', hover_data=['style', 'year'],
#    title=f"Mean-Shift Clustering - {selected_year}"
#)

fig = px.scatter(
    df_filtered,
    x='style_decoded',
    y='final_place',
    color=df_filtered['cluster'].astype(str),
    hover_data=['style', 'year'],
    title=f"Mean-Shift Clustering - {selected_year}",
    color_discrete_sequence=px.colors.qualitative.Set1,
)

fig.update_traces(marker=dict(size=10, opacity=0.7, line=dict(width=1, color='DarkSlateGrey')))
fig.update_layout(xaxis_title='Style', yaxis_title='Final Place')

fig.update_layout(
    xaxis=dict(tickangle=45),
    margin=dict(t=50, l=30, r=30, b=120)
)

st.plotly_chart(fig)

# --- Cluster Summary Stats ---
st.subheader("Cluster Summary Statistics")

# 1. Number of entries per cluster
cluster_counts = df_filtered['cluster'].value_counts().sort_index()

# 2. Mean final place per cluster
mean_final_place = df_filtered.groupby('cluster')['final_place'].mean()

# 3. Most common style per cluster
most_common_styles = df_filtered.groupby('cluster')['style'].agg(lambda x: x.value_counts().idxmax())

# Combined stats table
summary_df = pd.DataFrame({
    'Count': cluster_counts,
    'Avg Final Place': mean_final_place,
    'Most Common Style': most_common_styles
})

st.write("### Cluster Summary Table - per cluster")
st.dataframe(summary_df)

def show_clustering():
    st.write("")