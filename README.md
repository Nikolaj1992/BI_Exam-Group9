# Business Intelligence Exam 2025 - Group 9

Group members:
- Jenny Josephine Lenebjer Nielsen
- Patrick Alexander Kempf
- Nikolaj Fjellerad
- David Alexander Vig

Problem:
* The Eurovision Song Contest assigns each competing song a specific performance position (running order) during the show. Over time, fans and commentators have speculated that songs performed later tend to receive higher scores, potentially due to recency bias. There's also a belief that certain positions, such as the second slot (“death slot”), consistently lead to poor outcomes. This project aims to explore whether the running order, music style, and differences between jury and televote influence the final results, using historical data from Eurovision.

Idea:
* Our idea is to analyze Eurovision Song Contest data, over several years, in order to prove or disprove the following hypotheses.

Hypotheses:
- Hypothesis 1: There is a correlation between the running order (R/O) and the final result. Performing later tends to be associated with a better final result.
- Hypothesis 2: Contestants who perform in the last third of the show receive higher average scores than those performing in the first third of the show.
- Hypothesis 3: Slot 2 is often referred to as the “death slot” as no contestant performing at this position has ever won Eurovision, and consistently rank lower than other positions, so a song performing second will perform worse in the final result than it would otherwise.
- Hypothesis 4: Voting patterns differ between juries and the televote in regards to the running order. The televote appears more influenced by when a song is performed compared to the juries.
- Hypothesis 5: The choice of music style has a big impact on the final result.

Motivation:
* Our motivation is to use Eurovision data to better understand the contest and what impacts the results of the contest. A few members of the team regularly follow and watch Eurovision, and as such there's an interest in better understanding the outcomes of the contest with data.

Theoretical foundation:
* What???

Argumentation of choices:
* The following methodological and analytical choices were made based on the goals of the project and the nature of the available data.
Historical Data Analysis:
* Eurovision offers a rich and consistent dataset over decades. Analyzing historical data allows for trend identification over time, especially concerning running order and voting patterns.
* Division into Thirds (Early, Middle, Late): Breaking the performance order into thirds provides a simple but effective structure to compare early versus late performance outcomes. 
* Slot 2 Analysis ("Death Slot"): The unique attention given to position 2 by fans and analysts makes it worthy of special investigation. It’s a commonly discussed anomaly, and this study aims to verify if it’s statistically significant or merely a myth.
* Separate Analysis of Jury vs. Televote: The dual voting system enables an exploration of whether professional juries are less susceptible to performance order bias than the public. This comparison strengthens the analysis by offering a control variable.
* Inclusion of Music Style as a Variable: Since genre may play a role in audience reception, incorporating music style into the analysis helps isolate whether outcomes are due to performance timing or musical content. 

Design:
* Design is simple. Write code in notebook format, convert to .py files so that we can conveniently display our findings in an easy and user-friendly manner using Streamlit.

Outcomes:
* We've managed to prove a correlation between running order and final result (hypothesis 1), and that songs in the last third do in fact receive higher average scores (hypothesis 2). Analysing the infamous death slot also resulted in us being able to prove 2nd draw position to be the worst out of all (hypothesis 3). We could also find a difference in the voting patterns of juries and the televote, albeit not always quite necessarily the way we had anticipated (hypothesis 4).
* Lastly when it comes to music style we found that certain styles, like pop, appear far more often in the contest, but statistically we couldn't prove any big/meaningful impact on final results (hypothesis 5 - not proved).

## Implementation instructions
##### !pip install -r requirements.txt - run this in a notebook to install it all

Requirements:
- streamlit
- streamlit_folium
- streamlit-option-menu
- langchain_core
- langchain_community
- langchain_ollama
- selenium
- unstructured[pdf]
- yellowbrick
- pycountry
- kaleido
- scikit-learn
- matplotlib
- plotly
- scipy

System dependencies required (not installed via pip):
- poppler (for PDF parsing) #!conda install poppler -y
- tesseract (for OCR) #!conda install tesseract -y

Important notes:
##### Not all groupmembers have been able to get tesseract working
