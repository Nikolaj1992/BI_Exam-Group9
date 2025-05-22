import streamlit as st
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np, pandas as pd, seaborn as sbn, matplotlib.pyplot as plt, joblib
import plotly.express as px

from Modules import utils as utl
from Modules import data_exploration as de
from scipy.stats import zscore, ttest_ind

st.title("Death-Slot Analysis - 2nd Draw Position")
st.subheader("Let's see if the infamous 2nd draw position aka the 'Death Slot' is in fact as bad as the name suggests. No entry in Eurovision has ever won a final from this starting position, is that just chance?", divider='rainbow')

# Load data
csv_finals_path = '../Data/finalists_cleaned.csv'
finals_df = utl.load_csv(csv_finals_path, skiprows=0, encoding='windows-1252')

# Preprocessing data
finals_df = finals_df[finals_df['final_draw_position'] <= 26]

# Toggle to exclude outlier
exclude_outlier = st.checkbox("Exclude outlier (6th place in draw position 2)", value=False)

if exclude_outlier:
    # Remove the 6th-place outlier for position 2, dynamically changes page
    finals_df = finals_df[~((finals_df['final_draw_position'] == 2) & (finals_df['final_place'] == 6))]

# Year filtering
years = sorted(finals_df['year'].unique())
selected_years = st.slider("Select Year Range", min_value=int(min(years)), max_value=int(max(years)), value=(2009, 2023))
finals_df = finals_df[(finals_df['year'] >= selected_years[0]) & (finals_df['year'] <= selected_years[1])]

st.write("Feel free to adjust a year range to see data for only those years. Keep in mind however, that conclusions and the overall analysis is still based on the full dataset with all years included.")

st.markdown("""
#### EDA (Exploratory Data Analysis) - Does position 2 stand out?
""")
# Summary Statistics
draw_position_stats_finals = finals_df.groupby('final_draw_position')['final_place'].agg(['count', 'mean', 'median', 'std']).sort_index()
draw_position_stats_finals.loc[2]

# Boxplot & Barplot
st.subheader("Boxplot & Barplot of Final Place by Draw Position")
fig, ax = plt.subplots()
de.vs.boxplot(data=finals_df, x='final_draw_position', y='final_place', title='Final Place by Draw Position', 
              xlabel='Final Draw Position', ylabel='Final Place (Lower is Better)', fig=fig, ax=ax)
st.pyplot(fig)

fig2, ax2 = plt.subplots()
de.vs.barplot(finals_df, x_col='final_draw_position', y_col='final_place', agg_func='mean', 
              title='Average Final Place per Draw Position', 
              xlabel='Draw Position', ylabel='Average Final Place')
st.pyplot(fig2)

# Result Explanations
if exclude_outlier:
    st.markdown("""
##### From the finals of 2009–2023, outlier free:
- **13 entries** have performed from the 2nd draw position.
- **Mean/average placement:** 20.53 — poor performance, worse now.
- **Median placement:** 22 — worse than mean, performance skews downward.
- **Standard Deviation:** ±3.642 — mostly entries place between 17th and 24th.

From the boxplot we can tell that 2nd draw position has:
- Slightly more narrow variability (interquartile range - middle 50% of data goes from 17th to 23rd place)
- Still the worst average result of any draw position

The box for 2nd position itself is the second smallest one suggesting low variability in performance, while also being the most upward box - so far confirming our hypothesis that the Death-Slot is a real thing. Other draw positions have similarly high medians, with greater variability but none quite as profound as the 2nd draw position.

The barplot shows us the overall performance of each draw position, and it's clear that 2nd position has had the worst (mean/average) performance with a fairly low standard error. At the same time we see that draw positions 11 and 20 appear to be the, on average, best performing draw positions.
""")
else:
    st.markdown("""
##### From the finals of 2009–2023:
- **14 entries** have performed from the 2nd draw position.
- **Mean/average placement:** 19.5 — poor performance, not great.
- **Median placement:** 21.5 — worse than mean, performance skews downward.
- **Standard Deviation:** ±5.229 — mostly entries place between 14th and 25th.

From the boxplot we can tell that 2nd draw position has:
- Narrow variability (tight interquartile range - middle 50% of data goes from 16th to 23rd place)
- The whiskers are fairly narrow suggesting the data, outliers excluded, has limited range
- The worst average result of any draw position
- One outlier: a 6th place finish, best performance by far in the dataset

The box for 2nd position itself is the second smallest one suggesting low variability in performance, while also being the most upward box - so far confirming our hypothesis that the Death-Slot is a real thing. Other draw positions have similarly high medians, with greater variability but none quite as profound as the 2nd draw position.
The barplot shows us the overall performance of each draw position, and it's clear that 2nd position has had the worst (mean/average) performance with a fairly low standard error. At the same time we see that draw positions 11 and 20 appear to be the, on average, best performing draw positions. 
""")

st.markdown("""
##### Attempt at testing our hypothesis and use z-scores to see if position 2 is itself an outlier in performance
""")

st.latex(r"""
z = \frac{x - \mu}{\sigma}
""")
st.markdown("Where:  \n- $x$ is the individual value (mean placement of a draw position)  \n- $\\mu$ is the overall mean of the draw position means  \n- $\\sigma$ is the standard deviation of the draw position means")

# Z-Score Analysis
draw_position_means_finals = draw_position_stats_finals['mean']
z_scores_finals = zscore(draw_position_means_finals)
position_2_z_finals = z_scores_finals[draw_position_stats_finals.index == 2]

z_scores_finals

# Z-Score Explanations
if exclude_outlier:
    st.markdown("""
##### Computing z-scores of the mean final place for each draw position to see if 2 is "extreme"/stands out, no outlier.
- Z-score for draw position 2: **2.512**
- This means draw position 2's mean is over **2.5 standard deviations worse** than the average.
- It's statistically **an outlier** since (z > 1.96).

The new 2nd draw position z-score of 2.512 means that the average final place of entries starting at position 2 are over 2.5 standrad deviations worse than the overall average of all positions. Position 2 remains an outlier while position 20 does as well, albeit as the best draw position.
""")
else:
    st.markdown("""
##### Computing z-scores of the mean final place for each draw position to see if 2 is "extreme"/stands out.
- Z-score for draw position 2: **2.232**
- This means draw position 2's mean is over **2 standard deviations worse** than the average.
- It's statistically **an outlier** since (z > 1.96).

With 2nd draw position having a z-score of **2.232** it means that the average final place of entries starting at position 2 are over 2 standrad deviations worse than the overall average of all positions.
With 1.96 typically being enough to be considered statistically "extreme", we have exceeded this number. This means we can technically say that 2nd draw position is an outlier by itself, it is clear that we have a strong indicator that Position 2 has one of the worst average placements. No other draw positions has a higher z-score than position 2. Interestingly, we do see that position 20 has a z-score of -2.032 also making it an outlier in itself but as being possibly the best draw position to have.
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

# T-Test - Testing our hypothesis and using z-scores to see if position 2 is itself an outlier in performance
position_2_finals = finals_df[finals_df['final_draw_position'] == 2]['final_place']
other_positions_finals = finals_df[finals_df['final_draw_position'] != 2]['final_place']
t_stat, p_val = ttest_ind(position_2_finals, other_positions_finals, equal_var=False)
st.write(f"T-statistic: {t_stat}, p-value: {p_val}")

# T-Test Explanations
if exclude_outlier:
    st.markdown("""
##### T-Test of position 2 against all others, no outlier
- T-statistic: **6.78**
- P-value: **0.000004**
- This even lower p-value also allows us to **reject the null hypothesis**, confirming our original hypothesis that position 2 performs significantly worse than others.

Using Welch's t-test again, comparing position 2 versus all other draw positions, we get a new smaller p-value of 0.000004 and a t-statistic of 6.78. The t-statistic indicates that the difference between the mean final placements of these two groups is approx. 6.8 standard errors apart, which is strong evidence of a meaningful difference. 
Our result when comparing position 2 to all other positions combined, is that we observe significantly worse average placement for postion 2. This must be taken with a slight grain of salt however as we are comparing entries at position 2 with entries from all other positions - creating an imbalance as we do know other draw positions have similar, if not as profound, disadvantages. However, this score is still significant enough that even compared against a field of positions, that we can say it is a worse draw position.
The new p-value is even lower. Our hypothesis being that starting at 2nd draw position aka the Death-Slot is signifanctly worse than any other draw position, this means our null hypothesis is that there is no difference. If the p-value is smaller than 0.05 that's ordinarily enough to reject the null hypothesis. At just 0.000004 we can easily reject the null hypothesis again - strongly suggesting that the 2nd position does in fact perform statistically worse. Our original hypothesis — that position 2 performs significantly worse — is well supported here.
""")
else:
    st.markdown("""
##### T-Test of position 2 against all others
- T-statistic: **4.353**
- P-value: **0.0005**
- This very low p-value allows us to **reject the null hypothesis**, confirming our original hypothesis that position 2 performs significantly worse than others.

Using Welch's t-test, comparing position 2 versus all other draw positions, we've gotten our p-value and t-statistic.
The t-statistic indicates that the difference between the mean final placements of these two groups is approx. 4.4 standard errors apart, which is strong evidence of a meaningful difference. 
Our result when comparing position 2 to all other positions combined, is that we observe significantly worse average placement for postion 2. This must be taken with a slight grain of salt however as we are comparing entries at position 2 with entries from all other positions - creating an imbalance as we do know other draw positions have similar, if not as profound, disadvantages. However, this score is still significant enough that even compared against a field of positions, that we can say it is a worse draw position.
Our hypothesis - that starting at 2nd draw position aka the Death-Slot is signifanctly worse than any other draw position, means that our null hypothesis is that there is no difference. If the p-value is smaller than 0.05 that's ordinarily enough to reject the null hypothesis. Ours is much lower at just 0.0005, meaning that it is highly unlikely that entries starting at the 2nd draw position have performed badly by mere chance alone - strongly suggesting that the 2nd position does in fact performe statistically worse.
""")