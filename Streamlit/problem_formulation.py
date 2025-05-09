import streamlit as st

def show_problem_formulation():
    st.title("📘 Problem Formulation")

    st.markdown("""
    The **Eurovision Song Contest** attracts more than 160 million viewers every year across Europe and beyond.  
    Every competing song is performed from a specific position — referred to as the **running order**.  

    Fans and journalists have long speculated that performing later in the show might increase a song’s chances of success.  
    This project tests whether this recency bias actually exists by analyzing historical data from the Eurovision Song Contest.
    """)

    st.header("🧪 Research Questions")

    st.write("""
    - Does the running order of performances influence the final ranking or score of a song?
    - How does performance timing (early vs. late) correlate with the final result?
    - Is there a statistically significant difference in rankings between early and late performances?
    - Does the so-called "death slot" (position 2) lead to worse results?
    - Do jury and televote patterns differ in how they respond to running order?
    - How does increasing viewership throughout the show impact results?
    """)

    st.header("📌 Hypotheses")

    st.write("""
    - **H1**: There is a correlation between running order and final result — later performers tend to do better.
    - **H2**: Songs in the last third of the show receive higher average scores than those in the first third.
    - **H3**: The "death slot" (position 2) results in lower rankings and no winners.
    - **H4**: Televotes are more influenced by running order than jury votes, partially due to rising viewership as the show progresses.
    """)

    st.header("💡 Expected Solution")

    st.markdown("""
    We expect to explore this using data analysis tools like:
    - **Linear regression**
    - **Correlation analysis**
    - **Heatmaps**
    - **Scatter plots**

    We'll use these to explore:
    - Whether a meaningful correlation exists between running order and placement
    - The impact of other factors like genre and viewership changes
    """)

    st.header("🌍 Possible Impact")

    st.write("""
    - **Organizers**: Could use results to improve fairness in assigning running order.
    - **Fans & journalists**: Gain deeper understanding of performance biases.
    - **Betting sites**: Improve prediction models and odds-making.
    - **Analysts & academics**: Study the role of bias in international competitions and media events.
    """)
