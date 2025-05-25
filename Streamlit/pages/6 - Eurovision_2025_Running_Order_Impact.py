import pandas as pd
import numpy as np
import seaborn as sbn
import matplotlib.pyplot as plt
import streamlit as st

# Creating a dictionary with Eurovision 2025 data
csv_file_path = '../Data/eurovision_2025_grand_final.csv'
df = pd.read_csv(csv_file_path, encoding='windows-1252')

# Display the first 5 rows
st.write("# Eurovision 2025 Grand Final Results")
st.dataframe(df)

# Dropped columns (non-numeric data)
dropped_columns = ['country', 'year']
st.write("### Dropped Columns")
st.write("""
We have removed the following columns from the analysis:
- **'country'**: Not relevant to our question about performance order and voting results.
- **'year'**: The year of the contest, which we don't need for this analysis.
""")

# Dropping non-numeric columns before performing analysis
df_numerical = df.drop(columns=dropped_columns)

# ### Data Visualization

# Scatter plot of Performance Order vs Jury Points
st.write("# Performance Order vs Jury Points")
fig, ax = plt.subplots(figsize=(10, 6))
sbn.scatterplot(data=df_numerical, x='final_draw_position', y='final_jury_points', ax=ax)
plt.title('Performance Order vs Jury Points')
plt.xlabel('Performance Order (Final Draw Position)')
plt.ylabel('Jury Points')
plt.tight_layout()
st.pyplot(fig)

# Scatter plot of Performance Order vs Televote Points
st.write("# Performance Order vs Televote Points")
fig, ax = plt.subplots(figsize=(10, 6))
sbn.scatterplot(data=df_numerical, x='final_draw_position', y='final_televote_points', ax=ax)
plt.title('Performance Order vs Televote Points')
plt.xlabel('Performance Order (Final Draw Position)')
plt.ylabel('Televote Points')
plt.tight_layout()
st.pyplot(fig)

# ### Correlation Analysis
st.write("# Correlation Between Performance Order and Points")

# Correlation between Performance Order and Jury Points
jury_corr = df_numerical[['final_draw_position', 'final_jury_points']].corr()
st.write("### Correlation Table: Jury Points vs Performance Order")
st.write(jury_corr)

# Correlation between Performance Order and Televote Points
televote_corr = df_numerical[['final_draw_position', 'final_televote_points']].corr()
st.write("### Correlation Table: Televote Points vs Performance Order")
st.write(televote_corr)

# Heatmap for the correlation of all numeric columns
st.write("### Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10, 6))
sbn.heatmap(df_numerical.corr(), annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# ### Linear Regression: Jury Points vs Final Draw Position
st.write("# Linear Regression: Jury Points vs Final Draw Position")
fig, ax = plt.subplots(figsize=(10, 6))
sbn.regplot(x='final_draw_position', y='final_jury_points', data=df_numerical, ax=ax, line_kws={"color": "green"})
plt.title('Linear Regression: Performance Order vs Jury Points')
plt.xlabel('Performance Order (Final Draw Position)')
plt.ylabel('Jury Points')
plt.tight_layout()
st.pyplot(fig)

# ### Linear Regression: Televote Points vs Final Draw Position
st.write("# Linear Regression: Televote Points vs Final Draw Position")
fig, ax = plt.subplots(figsize=(10, 6))
sbn.regplot(x='final_draw_position', y='final_televote_points', data=df_numerical, ax=ax, line_kws={"color": "green"})
plt.title('Linear Regression: Performance Order vs Televote Points')
plt.xlabel('Performance Order (Final Draw Position)')
plt.ylabel('Televote Points')
plt.tight_layout()
st.pyplot(fig)

# ### Commentary on Results
st.write("# Commentary on Results")
st.write("""
#### Key Findings:
- **Jury Points**: From the linear regression analysis, the jury seems to have a **positive correlation** with performance order. Later-performing songs tend to receive more jury points.
- **Televote Points**: In contrast, televote points show a **negative correlation** with performance order. Earlier-performing songs received more televote points, with a significant drop as performance order increases.

#### Detailed Analysis of Results:

So from this data and the linear regression plots, it appears that this year **juries gave more points to songs later in the running order** (0.143012 = 14.3% more points for every song that has performed before you), while **televote gave more points to songs earlier in the running order** (-0.219494 = 21.2% fewer points for every song that has performed before you).

This is a clear **contrast to the general Jury vs Televote running order analysis**, where we typically observe televote favoring later songs. In this case, **televote points favored earlier songs**, indicating that the running order effect is **not always consistent across years**. 

Moreover, **the televote's top 2 songs** in this year performed as **number 3 and 4**, which **skewed the results** towards more points for an earlier running order. Therefore, the usual televote pattern of favoring later performers doesn't hold up this year due to these exceptions.

This deviation from the norm suggests that other factors, such as the **specific performance quality** or **public sentiment** for certain countries, could have influenced these results more strongly than just the performance order.
""")
