import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import f_oneway

# Load data
df = pd.read_csv('../Data/finalists_cleaned.csv')
df = df[['style', 'final_place']].dropna()
df['is_winner'] = (df['final_place'] == 1).astype(int)

# Page config
st.set_page_config(page_title="Eurovision Genre Impact", layout="wide")

# Title
st.title("🎤Does Music Style Affect Final Placement?")
st.subheader("Initial Thoughts vs Statistics: Does Pop Really Win More?", divider='rainbow')

# Value counts
style_counts = df['style'].value_counts()
df['style_with_count'] = df['style'].apply(lambda x: f"{x} (n={style_counts[x]})")

# Boxplot
st.subheader("🎼 Final Placements by Style")
fig1, ax1 = plt.subplots(figsize=(10, 5))
sns.boxplot(data=df, x='style_with_count', y='final_place', ax=ax1)
plt.xticks(rotation=45)
st.pyplot(fig1)

# Win rate bar chart
win_rates = df.groupby('style')['is_winner'].mean().sort_values(ascending=False)
win_rates_df = win_rates.reset_index()
win_rates_df.columns = ['Style', 'Win Rate']
win_rates_df['Win Rate'] = win_rates_df['Win Rate'] * 100  # to percent

st.subheader("🏆 Win Rate by Style (%)")
fig2, ax2 = plt.subplots(figsize=(8, 5))
sns.barplot(data=win_rates_df, x='Style', y='Win Rate', hue='Style', palette='viridis', legend=False)
for i, rate in enumerate(win_rates_df['Win Rate']):
    plt.text(i, rate + 0.5, f"{rate:.2f}%", ha='center')
plt.ylabel("Win Rate (%)")
plt.xlabel("Style")
st.pyplot(fig2)

# ANOVA Test
grouped = [group['final_place'].values for _, group in df.groupby('style')]
f_stat, p_val = f_oneway(*grouped)

st.subheader("📊 Analysis of Variance Test Results")
st.markdown(f"**F-statistic:** {f_stat:.2f}  \n**p-value:** {p_val:.4f}")

if p_val < 0.05:
    st.success("✅ There is a statistically significant difference between genres.")
else:
    st.warning("❌ No statistically significant difference found between genres.")
st.markdown("""
### 🎯 Statistical Conclusion

✅ **F-statistic** is **low**, meaning the average final placements across genres are quite similar.  
🔍 **p-value** is **high (0.4528)**, which suggests the differences are likely due to **random noise**, not a real effect.

👉 **Conclusion:**  
There is **no strong statistical evidence** that the *style/genre* of a song significantly impacts the final result.
""")

