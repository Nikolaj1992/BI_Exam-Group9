import streamlit as st
import sys
import os

# So we can correctly locate data and also our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np, pandas as pd, seaborn as sbn, matplotlib.pyplot as plt, joblib
import plotly.express as px

from Modules import utils as utl
from Modules import data_exploration as de
from scipy.stats import zscore, ttest_ind

st.title("Death-Slot Analysis - Draw Positions Semi-Finals")
st.subheader("Let's see if the infamous 2nd draw position aka 'Death Slot' also applies to the semi-finals.", divider='rainbow')

st.write("**Hypothesis 3:** Slot 2 is often referred to as the “death slot” as no contestant performing at this position has ever won Eurovision, and consistently rank lower than other positions, so a song performing second will perform worse in the final result than it would otherwise.")

# Load data
csv_semis_path = '../Data/semifinalists_cleaned.csv'
semis_df = utl.load_csv(csv_semis_path, skiprows=0, encoding='windows-1252')

# Year filtering
years = sorted(semis_df['year'].unique())
selected_years = st.slider("Select Year Range", min_value=int(min(years)), max_value=int(max(years)), value=(2009, 2023))
semis_df = semis_df[(semis_df['year'] >= selected_years[0]) & (semis_df['year'] <= selected_years[1])]

st.write("Feel free to adjust a year range to see data for only those years. Keep in mind however, that conclusions and the overall analysis is still based on the full dataset with all years included.")

st.markdown("""
#### EDA (Exploratory Data Analysis) - Do any draw positions stand out?
""")
# Summary Statistics
draw_position_stats_semis = semis_df.groupby('semi_draw_position')['semi_place'].agg(['count', 'mean', 'median', 'std']).sort_index()
draw_position_stats_semis

st.write("Not every semi-final has the same number of contestants. The 18th and 19th draw positions have significantly fever entries - thus we filter out those draw positions due to too few observations, which could negatively impact our analysis.")
semis_df = semis_df[semis_df['semi_draw_position'] <= 17]
draw_position_stats_semis = semis_df.groupby('semi_draw_position')['semi_place'].agg(['count', 'mean', 'median', 'std']).sort_index()

# 3D Scatter Plot
st.plotly_chart(
    de.vs.scatter_3d(
        semis_df,
        x='semi_draw_position', y='year', z='semi_place',
        color='semi_place',
        title='3D Scatter: Draw Position vs Year vs Semi-Final Place',
        xlabel='Semi Draw Position',
        ylabel='Year',
        zlabel='Semi-Final Place',
        hover_data=['country', 'style']
    )
)

st.write("This scatter plot lets us view our data without the last two draw positions. Now we can continue with identifying if any draw positions are worse than others, and if the 2nd position is the worst one again.")

# Boxplot & Barplot
st.subheader("Boxplot & Barplot of Semi Place by Draw Position")
fig, ax = plt.subplots()
de.vs.boxplot(data=semis_df, x='semi_draw_position', y='semi_place', title='Semi-Final Place by Draw Position', 
              xlabel='Semi-Final Draw Position', ylabel='Semi-Final Place (Lower is Better)', fig=fig, ax=ax)
st.pyplot(fig)

fig2, ax2 = plt.subplots()
de.vs.barplot(semis_df, x_col='semi_draw_position', y_col='semi_place', agg_func='mean', 
              title='Average Semi-Final Place per Draw Position', 
              xlabel='Draw Position', ylabel='Average Semi Place')
st.pyplot(fig2)

# Result Explanations
st.markdown("""
##### From our draw position stats and our boxplot and barplot we now see the following:
- On average the **mean** of the **17 draw positions** seem to be higher early on and drop off towards the last half - the exception being 11th position.
- The **median** appears higher in the earlier entries and gradually dropping towards the latter half of the semi-finals.
- The **STD (standard deviation)** across our 17 draw positions seem quite consistent between **4.02 & 5.59**.

The boxplot shows that interestingly there's a fair amount of consistency. Both in terms of maximum and minimum data values but also the medians. The most **obvious exceptions are the 2nd and 11th draw positions**. We can see that **2nd position** has an **unusually high minimum data value** while 11th position has rather narrow high and low quartile values.
The **3rd and 17th positions** are also a little interesting in that, the former has the highest median and the latter has the lowest.

The barplot shows us the overall performance of each draw position, and it's clear that **2nd, 3rd and 11th positions have the 'worst' (mean/average) performance** with roughly equal standard errors. At the same time we see that **draw positions 6, 9 and 17 appear to be the, on average, best performing** draw positions. 
""")

st.markdown("""
#### Attempt at testing our hypothesis and use z-scores - Are certain positions noticeably more favourable/unfavourable than others ie. Is the 2nd position still "the death slot"?
""")

st.latex(r"""
z = \frac{x - \mu}{\sigma}
""")
st.markdown("Where:  \n- $x$ is the individual value (mean placement of a draw position)  \n- $\\mu$ is the overall mean of the draw position means  \n- $\\sigma$ is the standard deviation of the draw position means")

# Z-Score Analysis
draw_position_means_semis = draw_position_stats_semis['mean']
z_scores_semis = zscore(draw_position_means_semis)

z_scores_semis

st.write("To better visualize our Z-Scores, and how each draw position deviates from the overall average performance, we've plotted the Z-scores into a heatmap with a color gradient. Our color gradient (warm to cold) highlights these deviations. Positive Z-scores in red indicate draw positions that, on average, result in worse final places, while negative Z-scores in blue indicate draw positions that result in better final places. Thus this heatmap makes it clear which draw positions tend to underperform or outperform - relative to the average.")

# Z-Score Heatmap
grouped_means = semis_df.groupby('semi_draw_position')['semi_place'].mean()
z_scores = pd.DataFrame({
    'Draw Position': grouped_means.index,
    'Z-Score': zscore(grouped_means)
}).set_index('Draw Position').T
st.plotly_chart(de.vs.zscore_heatmap(z_scores, title='Z-Score Heatmap of Mean Semi-Final Places by Draw Position'))

st.write("A Z-Score between ±1.96 has a 95% confidence level, meaning there's only a 5% chance we got this result by random chance. This also means that any Z-Scores outside this range are so far from the mean, whilst there being a less than 5% chance of it being random, that we can say those are statsitical outliers.")

# Z-Score Explanations
st.markdown("""
##### Computing z-scores of the mean semi place for each draw position to see if any stands out.
- **2nd draw position** having a z-score of **1.937** it means that the average semi place of entries starting at position 2 are **just below 2 standrad deviations** worse than the overall average of all positions.
- **3rd draw position** having a z-score of **2.082** it means that the average semi place of entries starting at position 3 are **over 2 standrad deviations** worse than the overall average of all positions.
- **11th draw position** having a z-score of **1.532** it means that the average semi place of entries starting at position 11 are **1.5 standrad deviations** worse than the overall average of all positions.

With ±1.96 typically being enough to be considered statistically "extreme", we have exceeded this number once. This means **we can technically say that 3rd draw position is an outlier by itself**. No other draw positions has a higher z-score than position 3, albeit 2 is close. Interestingly, we do see that positions 9 and 17 have z-scores of -0.927 and -1.071 seemingly making them the best draw positions to have.
""")

st.latex(r"""
t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}
""")
st.markdown("Where:  \n- $\\bar{x}_1$ and $\\bar{x}_2$ are the sample means  \n- $s_1^2$ and $s_2^2$ are the sample variances  \n- $n_1$ and $n_2$ are the sample sizes")

st.latex(r"""
\text{p-value} = 2 \cdot P(T > |t|)
""")
st.markdown("""
Where:  
- $T$ is the t-distribution under the null hypothesis  
- $t$ is the absolute value of the observed t-statistic  
- The p-value is the probability of observing a test statistic as extreme as $t$, assuming the null hypothesis is true
""")

st.write("Worth investigating further are the 2nd and 3rd positions, given the z-scores we got. Thus we'll do a Welch's T-test.")

# T-Test - Testing our hypothesis and using z-scores - Worth investigating further are 2nd and 3rd positions
position_2_semis = semis_df[semis_df['semi_draw_position'] == 2]['semi_place']
other_positions_than_2_semis = semis_df[semis_df['semi_draw_position'] != 2]['semi_place']
position_3_semis = semis_df[semis_df['semi_draw_position'] == 3]['semi_place']
other_positions_than_3_semis = semis_df[semis_df['semi_draw_position'] != 3]['semi_place']

t_stat_2, p_val_2 = ttest_ind(position_2_semis, other_positions_than_2_semis, equal_var=False)
t_stat_3, p_val_3 = ttest_ind(position_3_semis, other_positions_than_3_semis, equal_var=False)

st.write(f"T-statistic - 2nd draw: {t_stat_2}, p-value: {p_val_2}")
st.write(f"T-statistic - 3rd draw: {t_stat_3}, p-value: {p_val_3}")

# T-Test Explanations
st.markdown("""
##### T-Test of positions 2 and 3
Draw position 2:
- T-statistic: **3.159**
- P-value: **0.003**
- This low p-value allows us to **reject the null hypothesis**, confirming our original hypothesis, as written at the top of the page, that **position 2 performs worse than other draw positions**, and not just due to random chance.

Draw position 3:
- T-statistic: **2.617**
- P-value: **0.013**
- This p-value, while not as low, means we can also **reject the null hypothesis**, in this case our hypothesis might simply be "That certain positions perform worse than others". There's enough evidence to support that position 3 is unfavourable, but not to same extent as that infamous position 2.

The t-statistics indicate that the difference between the mean semi placements of the two groups (X place vs rest of data) is approx. 3.15 and 2.61 standard errors apart respectively, which is strong evidence of a meaningful difference for both placements. 
Our result when comparing position 2 and 3 to all other positions combined, is that we observe worse average placement for postion 2 and 3. This must be taken with a slight grain of salt however as we are comparing with entries from all other positions - creating an imbalance. However, these scores are still significant enough that even compared against a field of positions, that we can say they are worse draw positions.

To summarize:
The p-values are low. If the p-value is smaller than 0.05 that's ordinarily enough to reject the null hypothesis. 
For 2nd position we can reject our null hypothesis.
For 3rd position we can reject our null hypothesis.
The 2nd position, like in the finals, continues to be a "death slot".
""")