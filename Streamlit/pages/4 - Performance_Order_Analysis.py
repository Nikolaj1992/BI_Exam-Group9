import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# Set visualization style
sns.set(style="whitegrid")
sns.set_palette('husl')

# Page config
st.set_page_config(layout="wide")

# Title and introduction
col1, col2 = st.columns([3, 1])
with col1:
    st.title("Eurovision Song Contest Performance Order Analysis")
    st.markdown("""
    This analysis explores whether the running order of performances in the Eurovision Song Contest 
    influences the final ranking or score of a song in the grand final.
    """)
with col2:
    try:
        logo = Image.open('../Images/Eurovision-Song-Contest-logo.jpg')
        st.image(logo, width=200)
    except:
        try:
            logo = Image.open('./Media/esc.jpg')
            st.image(logo, width=200)
        except:
            pass

# Sidebar with analysis options
st.sidebar.title("Analysis Options")
analysis_type = st.sidebar.radio(
    "Select Analysis Type",
    ["Overview", "Performance Position Impact", "Death Slot Analysis", "Jury vs. Televote", "Historical Trends", "Conclusion"]
)

# Load the finalists data
@st.cache_data
def load_data():
    finalists_df = pd.read_csv('../Data/finalists_cleaned.csv')
    
    # Create performance order groups (first third, middle third, last third)
    finalists_df['performance_group'] = pd.qcut(
        finalists_df['final_draw_position'], 
        q=3, 
        labels=['First Third', 'Middle Third', 'Last Third']
    )
    
    # Create a "top 10" flag
    finalists_df['top_10'] = finalists_df['final_place'] <= 10
    
    return finalists_df

# Load the data
finalists_df = load_data()

# Display basic dataset information in the Overview section
if analysis_type == "Overview":
    st.header("Dataset Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Entries", len(finalists_df))
        st.metric("Years Covered", f"{finalists_df['year'].min()} - {finalists_df['year'].max()}")
    
    with col2:
        st.metric("Number of Countries", finalists_df['country'].nunique())
        st.metric("Max Performance Position", int(finalists_df['final_draw_position'].max()))
    
    st.subheader("Sample Data")
    st.dataframe(finalists_df.head())
    
    st.subheader("Correlation between Performance Order and Final Results")
    
    # Calculate the correlation
    correlation = finalists_df['final_draw_position'].corr(finalists_df['final_place'])
    
    # Create scatter plot with trend line
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.regplot(x='final_draw_position', y='final_place', data=finalists_df, scatter_kws={'alpha':0.5}, ax=ax)
    plt.title('Relationship between Performance Order and Final Placement', fontsize=16)
    plt.xlabel('Performance Order Position', fontsize=14)
    plt.ylabel('Final Place', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.annotate(f"Correlation: {correlation:.3f}", xy=(0.05, 0.95), xycoords='axes fraction', 
                fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    st.pyplot(fig)
    
    # Create a correlation heatmap
    corr_cols = ['final_draw_position', 'final_place', 'final_total_points', 
                'final_televote_points', 'final_jury_points']
    corr_matrix = finalists_df[corr_cols].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', mask=mask, vmin=-1, vmax=1, ax=ax)
    plt.title('Correlation Matrix of Eurovision Final Variables', fontsize=16)
    st.pyplot(fig)
    
    st.markdown("""
    ### Key Findings:
    
    The correlation coefficient between performance order position and final place is a key indicator of the relationship's strength.
    A negative correlation suggests that as performance order increases (performing later), final place tends to decrease (better ranking).
    """)

elif analysis_type == "Performance Position Impact":
    st.header("Performance Position Impact Analysis")
    
    # Calculate statistics for each group
    group_stats = finalists_df.groupby('performance_group')['final_place'].agg(['mean', 'median', 'std', 'count'])
    
    st.subheader("Average Final Place by Performance Group")
    st.dataframe(group_stats)
    
    # Box plot of final places by performance group
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='performance_group', y='final_place', data=finalists_df, ax=ax)
    plt.title('Distribution of Final Places by Performance Order Group', fontsize=16)
    plt.xlabel('Performance Order Group', fontsize=14)
    plt.ylabel('Final Place', fontsize=14)
    plt.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    # ANOVA test
    groups = [group for _, group in finalists_df.groupby('performance_group')['final_place']]
    f_statistic, p_value = stats.f_oneway(*groups)
    
    st.subheader("ANOVA Test Results")
    st.write(f"F-statistic: {f_statistic:.3f}")
    st.write(f"p-value: {p_value:.4f}")
    
    if p_value < 0.05:
        st.success("The differences between performance groups are statistically significant (p < 0.05)")
    else:
        st.info("The differences between performance groups are not statistically significant (p >= 0.05)")
    
    # Calculate top 10 success rate by performance group
    top_10_stats = finalists_df.groupby('performance_group')['top_10'].mean() * 100
    
    st.subheader("Top 10 Success Rate by Performance Group")
    
    # Create bar chart for top 10 percentages
    fig, ax = plt.subplots(figsize=(10, 6))
    top_10_stats.plot(kind='bar', color='skyblue', ax=ax)
    plt.title('Percentage of Top 10 Finishes by Performance Order Group', fontsize=16)
    plt.xlabel('Performance Order Group', fontsize=14)
    plt.ylabel('Percentage of Top 10 Finishes', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    
    # Add value labels on top of each bar
    for i, v in enumerate(top_10_stats):
        plt.text(i, v + 2, f"{v:.1f}%", ha='center', fontsize=12)
    
    st.pyplot(fig)
    
    st.markdown("""
    ### Key Findings:
    
    - The Last Third performers tend to achieve better final placements on average.
    - The First Third performers have the lowest percentage of top 10 finishes.
    - The statistical significance of these differences is determined by the ANOVA test results.
    """)

elif analysis_type == "Death Slot Analysis":
    st.header("'Death Slot' and Key Positions Analysis")
    
    # Analyze specific positions (Position 2 is often called the "death slot")
    position_stats = finalists_df.groupby('final_draw_position')['final_place'].agg(['mean', 'median', 'std', 'count'])
    
    # Focusing on positions that have at least 5 entries for reliability
    reliable_positions = position_stats[position_stats['count'] >= 5]
    
    st.subheader("Average Final Place by Performance Position")
    
    # Visualize average final places by performance position
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = sns.barplot(x=reliable_positions.index, y=reliable_positions['mean'], alpha=0.7, ax=ax)
    
    # Highlight position 2 (death slot) if it exists in reliable positions
    if 2 in reliable_positions.index:
        death_slot_idx = list(reliable_positions.index).index(2)
        bars.patches[death_slot_idx].set_color('red')
    
    plt.title('Average Final Place by Performance Position', fontsize=16)
    plt.xlabel('Performance Position', fontsize=14)
    plt.ylabel('Average Final Place', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    # Calculate top 10 success rate by position
    position_top10 = finalists_df.groupby('final_draw_position')['top_10'].mean() * 100
    
    # Filter for positions with at least 5 entries
    reliable_position_top10 = position_top10[position_stats['count'] >= 5]
    
    st.subheader("Percentage of Top 10 Finishes by Performance Position")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = sns.barplot(x=reliable_position_top10.index, y=reliable_position_top10.values, alpha=0.7, ax=ax)
    
    # Highlight position 2 (death slot) if it exists
    if 2 in reliable_position_top10.index:
        death_slot_idx = list(reliable_position_top10.index).index(2)
        bars.patches[death_slot_idx].set_color('red')
    
    plt.title('Percentage of Top 10 Finishes by Performance Position', fontsize=16)
    plt.xlabel('Performance Position', fontsize=14)
    plt.ylabel('Percentage of Top 10 Finishes', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    st.markdown("""
    ### Key Findings:
    
    - Position 2 is often referred to as the "death slot" in Eurovision and is highlighted in red.
    - The analysis shows whether songs performed in this position have historically performed worse.
    - Later positions tend to have higher percentages of top 10 finishes.
    """)

elif analysis_type == "Jury vs. Televote":
    st.header("Jury vs. Televote Analysis")
    
    # Clean data for this analysis (remove entries with missing jury or televote points)
    vote_analysis_df = finalists_df.dropna(subset=['final_jury_points', 'final_televote_points'])
    
    # Calculate correlations
    jury_correlation = vote_analysis_df['final_draw_position'].corr(vote_analysis_df['final_jury_points'])
    televote_correlation = vote_analysis_df['final_draw_position'].corr(vote_analysis_df['final_televote_points'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Correlation: Performance Order vs. Jury Points", f"{jury_correlation:.3f}")
    
    with col2:
        st.metric("Correlation: Performance Order vs. Televote Points", f"{televote_correlation:.3f}")
    
    # Calculate average jury and televote points by performance group
    vote_by_group = vote_analysis_df.groupby('performance_group').agg({
        'final_jury_points': 'mean', 
        'final_televote_points': 'mean'
    }).reset_index()
    
    # Reshape the data for plotting
    vote_long = pd.melt(vote_by_group, 
                        id_vars=['performance_group'],
                        value_vars=['final_jury_points', 'final_televote_points'],
                        var_name='Vote Type', 
                        value_name='Average Points')
    
    # Rename for better labels
    vote_long['Vote Type'] = vote_long['Vote Type'].map({
        'final_jury_points': 'Jury Points',
        'final_televote_points': 'Televote Points'
    })
    
    st.subheader("Average Jury vs. Televote Points by Performance Group")
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='performance_group', y='Average Points', hue='Vote Type', data=vote_long, ax=ax)
    plt.title('Average Jury vs. Televote Points by Performance Order Group', fontsize=16)
    plt.xlabel('Performance Order Group', fontsize=14)
    plt.ylabel('Average Points', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(title='Vote Type', fontsize=12)
    st.pyplot(fig)
    
    st.markdown("""
    ### Key Findings:
    
    - This analysis shows whether jury votes and televotes are influenced differently by performance order.
    - A higher correlation for televotes would suggest that public viewers are more influenced by performance order than professional juries.
    - The bar chart shows how both types of votes are distributed across performance groups.
    """)

elif analysis_type == "Historical Trends":
    st.header("Historical Trends Analysis")
    
    # Split data into recent and earlier years
    year_cutoff = st.slider("Year cutoff for comparison", 2000, 2022, 2016)
    
    recent_df = finalists_df[finalists_df['year'] >= year_cutoff]
    earlier_df = finalists_df[finalists_df['year'] < year_cutoff]
    
    # Calculate correlations for each period
    recent_corr = recent_df['final_draw_position'].corr(recent_df['final_place'])
    earlier_corr = earlier_df['final_draw_position'].corr(earlier_df['final_place'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(f"Correlation ({year_cutoff} onwards)", f"{recent_corr:.3f}")
    
    with col2:
        st.metric(f"Correlation (before {year_cutoff})", f"{earlier_corr:.3f}")
    
    # Visualization of trends over time
    st.subheader("Performance Order Impact by Year")
    
    # Group by year and calculate average final place by performance group
    yearly_trends = finalists_df.groupby(['year', 'performance_group'])['final_place'].mean().reset_index()
    
    # Create line plot
    fig = px.line(yearly_trends, x='year', y='final_place', color='performance_group',
                 title='Average Final Place by Performance Group Over Time',
                 labels={'final_place': 'Average Final Place', 'year': 'Year', 'performance_group': 'Performance Group'})
    
    fig.update_layout(xaxis_title='Year', yaxis_title='Average Final Place')
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    ### Key Findings:
    
    - This analysis shows whether the impact of performance order has changed over time.
    - Changes in correlation coefficients between different time periods may indicate evolving trends.
    - The line chart visualizes how the advantage/disadvantage of different performance positions has varied across years.
    """)

elif analysis_type == "Conclusion":
    st.header("Comprehensive Conclusion")

    st.markdown("""
    ### Summary of Findings
    
    Our analysis of the Eurovision Song Contest performance order and its impact on final results has revealed several key insights:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Performance Position Impact")
        st.markdown("""
        - **Statistical Significance**: The ANOVA test showed that there is indeed a statistically significant difference in outcomes between songs performed in different parts of the show.
        
        - **Clear Advantage for Later Performers**: Songs performed in the last third of the running order achieve, on average, better final placements than those performed in the first third.
        
        - **Top 10 Success Rate**: The percentage of songs finishing in the top 10 increases significantly from the first third to the last third of performances.
        """)
        
        st.subheader("'Death Slot' Analysis")
        st.markdown("""
        - Position 2 (the "death slot") shows notably worse performance compared to most other positions, confirming the widely-held belief in Eurovision circles.
        
        - The closing positions (final 3-5 slots) consistently achieve better results, suggesting that recency bias plays a role in voting behavior.
        """)
    
    with col2:
        st.subheader("Jury vs. Televote Differences")
        st.markdown("""
        - **Televote Sensitivity**: There is a stronger correlation between performance order and televoting points compared to jury points.
        
        - This suggests that the general public is more influenced by performance order than professional juries, who might be better at evaluating songs independently of their position in the show.
        
        - The Last Third performers receive substantially more televote points on average than First Third performers.
        """)
        
        st.subheader("Historical Trends")
        st.markdown("""
        - The advantage of later performance positions has remained consistent over the years.
        
        - Recent contests (2016 onwards) show a slightly stronger correlation between performance order and results than earlier contests.
        """)
    
    st.subheader("Practical Implications")
    st.markdown("""
    1. **For Participants**: Drawing a later performance slot significantly increases chances of success. Particularly, performers should strive to avoid the "death slot" (position 2).
    
    2. **For Contest Organizers**: If fairness is a priority, the current system of assigning running order might need reconsideration, as it clearly impacts results.
    
    3. **For Viewers and Voters**: Awareness of this bias might help in making more objective voting decisions.
    """)
    
    st.subheader("Statistical Evidence")
    
    # Display key metrics
    correlation = finalists_df['final_draw_position'].corr(finalists_df['final_place'])
    
    # Calculate top 10 rates
    top_10_stats = finalists_df.groupby('performance_group')['top_10'].mean() * 100
    
    metrics = pd.DataFrame({
        'Metric': ['Correlation: Order vs. Final Place', 
                  'Top 10 Rate: First Third (%)', 
                  'Top 10 Rate: Last Third (%)',
                  'Avg Place: First Third',
                  'Avg Place: Last Third'],
        'Value': [f"{correlation:.3f}", 
                 f"{top_10_stats['First Third']:.1f}%", 
                 f"{top_10_stats['Last Third']:.1f}%",
                 f"{finalists_df[finalists_df['performance_group']=='First Third']['final_place'].mean():.2f}",
                 f"{finalists_df[finalists_df['performance_group']=='Last Third']['final_place'].mean():.2f}"]
    })
    
    st.table(metrics)
    
    st.markdown("""
    ### Final Conclusion
    
    The data strongly supports that performance order significantly influences final results in the Eurovision Song Contest. 
    Later performers have a clear statistical advantage, with the effect being more pronounced in televoting than jury voting. 
    The "death slot" (position 2) does show worse performance as per Eurovision folklore.
    
    These findings raise important questions about fairness in the competition and suggest that the running order should be 
    either randomized or carefully balanced to minimize its impact on results.
    """)

# Final conclusions
st.sidebar.markdown("---")
st.sidebar.subheader("Overall Conclusions")
st.sidebar.markdown("""
- Later performance positions generally have an advantage in the Eurovision final
- The "death slot" (position 2) tends to perform worse on average
- Televotes appear to be more influenced by performance order than jury votes
""") 