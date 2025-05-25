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
Hypothesis 5: The choice of music style has a big impact on the final result.

Motivation:
* Our motivation is to use Eurovision data to better understand the contest and what impacts the results of the contest. A few members of the team regularly follow and watch Eurovision, and as such there's an interest in better understanding the outcomes of the contest with data.

Theoretical foundation:
* What???

Argumentation of choices:
* More text needed....

Design:
* Design is simple. Write code in notebook format, convert to .py files so that we can convieniently display our findings in an easy and user-friendly manner using Streamlit.

Artefacts:
* hmmmmm x2

Outcomes:
* hmm


## Implementation instructions

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
##### Not all groupmembers have been able to get tesseract working.
##### Run !pip install -r requirements.txt for implementation.
