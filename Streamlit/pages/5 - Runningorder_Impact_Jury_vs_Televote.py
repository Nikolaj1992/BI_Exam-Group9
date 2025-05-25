import pandas as pd
import numpy as np
import seaborn as sbn
import matplotlib.pyplot as plt
import streamlit as st
import sys 
import os

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# So we can correctly locate data and also our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from Modules import utils as utl
from Modules import data_exploration as de
from Modules import machine_learning as ml
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import pickle
from scipy.spatial.distance import cdist

# ### Introduction to the Research Question 
st.write("# Research Question")
st.write("""
The goal of this analysis is to determine whether voting patterns differ between the jury and televote in terms of performance order, and if the televote is more influenced by performance timing. Specifically, we investigate:

1. **Do voting patterns differ between the jury and televote?**
2. **Is the televote more influenced by performance timing (final draw position)?**

In this analysis, we explore the relationship between the performance order (`final_draw_position`) and the points awarded by both the jury (`final_jury_points`) and televote (`final_televote_points`). Our goal is to understand how much of an effect the performance order (which could be understood as the running order of the performances in the Eurovision Song Contest) has on the final scores.

We analyze whether performance order influences voting behavior more for televotes than jury votes. A critical part of this analysis is determining how well a machine learning model can predict these voting patterns. For this purpose, we use different regression models, including **Linear Regression** and **Stacking Regressor**, to compare their predictive power.
""")

# ### Data Preprocessing
st.write("# Data Preprocessing")
csv_file_path = '../Data/finalists_cleaned.csv'
eurovision_df = pd.read_csv(csv_file_path, encoding='windows-1252')

# Dataset Overview
st.write("### Dataset Overview")
st.dataframe(eurovision_df)  # Display the entire DataFrame (if it's not too large)


# Dropped Columns Section
dropped_columns = ['style', 'final_televote_votes', 'final_jury_votes', 'country']
st.write("### Dropped Columns")
st.write("""
We have dropped several columns that were not deemed relevant for our analysis, including:
- **'style'**: Represents the style of music, which is not part of the analysis.
- **'final_televote_votes'**: The raw votes, which are not necessary once we focus on the points awarded.
- **'final_jury_votes'**: Similar to the televote votes, we work directly with the points awarded.
- **'country'**: We did not focus on country-specific analysis, so this column was removed.
""")
st.write("Dropped columns:", ', '.join(dropped_columns))

# Handle missing values with SimpleImputer
imputer = SimpleImputer(strategy='mean')
jury_televote_imputed = pd.DataFrame(imputer.fit_transform(eurovision_df.drop(columns=dropped_columns)), columns=eurovision_df.drop(columns=dropped_columns).columns)

# Handling missing values with imputation
st.write("### Handling Missing Values")
st.write("""
Missing values were handled by replacing them with the mean value of each respective column. This was done to ensure that no rows were lost in the analysis and that we could continue working with complete datasets.
""")
st.write("### Imputed Dataset")
st.dataframe(jury_televote_imputed.head())

# #### Exploring Correlations between Performance Order and Voting Results
st.write("# Correlation Between Performance Order and Points")
# Correlation between final draw position and both jury and televote points
jury_corr = jury_televote_imputed[['final_draw_position', 'final_jury_points']].corr()
televote_corr = jury_televote_imputed[['final_draw_position', 'final_televote_points']].corr()

st.write("### Correlation Table: Jury Points vs Performance Order")
st.write(jury_corr)

st.write("### Correlation Table: Televote Points vs Performance Order")
st.write(televote_corr)

# Create a correlation heatmap
st.write("### Correlation Heatmap")
st.write("""
The correlation heatmap shows the relationship between performance order (`final_draw_position`) and both the jury points (`final_jury_points`) and televote points (`final_televote_points`).
It provides insight into how strongly performance order influences the points awarded by the jury and televote.
""")
fig, ax = plt.subplots(figsize=(10, 6))
sbn.heatmap(jury_televote_imputed.corr(), annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# #### Visualize Data Distributions: Box Plots
st.write("# Visualizing Data Distributions with Box Plots")
# Jury Points vs Final Draw Position
fig, ax = plt.subplots(figsize=(12, 6))
sbn.boxplot(x='final_draw_position', y='final_jury_points', data=jury_televote_imputed, ax=ax)
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)

# Televote Points vs Final Draw Position
fig, ax = plt.subplots(figsize=(12, 6))
sbn.boxplot(x='final_draw_position', y='final_televote_points', data=jury_televote_imputed, ax=ax)
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)

# #### Linear Regression Models: Jury Points and Televote Points
st.write("# Linear Regression: Jury vs Televote Points")
# Scatter plot for Jury Points and Performance Order
fig, ax = plt.subplots(figsize=(10, 6))
sbn.scatterplot(data=jury_televote_imputed, x='final_draw_position', y='final_jury_points', ax=ax)
plt.title('Performance Order vs Jury Points')
plt.xlabel('Performance Order (Final Draw Position)')
plt.ylabel('Jury Points')
st.pyplot(fig)

# Scatter plot for Televote Points and Performance Order
fig, ax = plt.subplots(figsize=(10, 6))
sbn.scatterplot(data=jury_televote_imputed, x='final_draw_position', y='final_televote_points', ax=ax)
plt.title('Performance Order vs Televote Points')
plt.xlabel('Performance Order (Final Draw Position)')
plt.ylabel('Televote Points')
st.pyplot(fig)

# ** Linear Regression Model for Jury **

# ** Train Linear Regression model **
X = jury_televote_imputed.drop(columns=['final_jury_points', 'final_televote_points', 'final_total_points'])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Define X_scaled here
y_jury = jury_televote_imputed['final_jury_points']
y_televote = jury_televote_imputed['final_televote_points']

X_jury_train, X_jury_test, y_jury_train, y_jury_test = train_test_split(X_scaled, y_jury, test_size=0.2, random_state=42)
linear_reg_model_jury = LinearRegression()
linear_reg_model_jury.fit(X_jury_train, y_jury_train)
y_jury_pred_linear = linear_reg_model_jury.predict(X_jury_test)

# Calculate R² for Linear Regression
r2_jury = metrics.r2_score(y_jury_test, y_jury_pred_linear)
st.write(f"Linear Regression Model for Jury: R² value of **{r2_jury:.4f}**")

# ** Linear Regression Model for Televote **

# ** Train Linear Regression model **
X_televote_train, X_televote_test, y_televote_train, y_televote_test = train_test_split(X_scaled, y_televote, test_size=0.2, random_state=42)
linear_reg_model_televote = LinearRegression()
linear_reg_model_televote.fit(X_televote_train, y_televote_train)
y_televote_pred_linear = linear_reg_model_televote.predict(X_televote_test)

# Calculate R² for Linear Regression
r2_televote = metrics.r2_score(y_televote_test, y_televote_pred_linear)
st.write(f"Linear Regression Model for Televote: R² value of **{r2_televote:.4f}**")

# Data Normalization
st.write("# Data Normalization")
st.write("""
Normalization was applied using **StandardScaler** to ensure that each feature has a mean of 0 and a standard deviation of 1. This scaling process allows for better comparison of features and prevents features with larger scales from dominating the analysis.
""")

X = jury_televote_imputed.drop(columns=['final_jury_points', 'final_televote_points', 'final_total_points'])  # Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y_jury = jury_televote_imputed['final_jury_points']
y_televote = jury_televote_imputed['final_televote_points']

# ### Train and Test Model with Stacking Regressor
st.write("# Stacking Regressor: Combining Random Forest and Linear Regression")

# Jury Model: Stacking Regressor
X_jury_train, X_jury_test, y_jury_train, y_jury_test = train_test_split(X_scaled, y_jury)
base_learners = [
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
    ('lr', LinearRegression())
]
stacking_model_jury = StackingRegressor(estimators=base_learners, final_estimator=LinearRegression())
stacking_model_jury.fit(X_jury_train, y_jury_train)
y_jury_predicted = stacking_model_jury.predict(X_jury_test)

r2_stack_jury = metrics.r2_score(y_jury_test, y_jury_predicted)
st.write(f"Jury Model: R² value of **{r2_stack_jury:.4f}**, which is higher than the Linear Regression model's R².")

# Televote Model: Stacking Regressor
X_televote_train, X_televote_test, y_televote_train, y_televote_test = train_test_split(X_scaled, y_televote)
stacking_model_televote = StackingRegressor(estimators=base_learners, final_estimator=LinearRegression())
stacking_model_televote.fit(X_televote_train, y_televote_train)
y_televote_predicted = stacking_model_televote.predict(X_televote_test)

r2_stack_televote = metrics.r2_score(y_televote_test, y_televote_predicted)
st.write(f"Televote Model: R² value of **{r2_stack_televote:.4f}**, which shows a stronger correlation with performance order, significantly higher than the simple linear regression model.")

# ### Margin of Errors for the Stacking Model
st.write("# Margin of Errors for the Stacking Model")
# Jury Error Margin
st.write("### Jury Model Errors")
st.write(f"Mean Absolute Error (MAE) = {metrics.mean_absolute_error(y_jury_test, y_jury_predicted):.4f}")
st.write(f"Mean Squared Error (MSE) = {metrics.mean_squared_error(y_jury_test, y_jury_predicted):.4f}")
st.write(f"Root Mean Squared Error (RMSE) = {np.sqrt(metrics.mean_squared_error(y_jury_test, y_jury_predicted)):.4f}")

# Televote Error Margin
st.write("### Televote Model Errors")
st.write(f"Mean Absolute Error (MAE) = {metrics.mean_absolute_error(y_televote_test, y_televote_predicted):.4f}")
st.write(f"Mean Squared Error (MSE) = {metrics.mean_squared_error(y_televote_test, y_televote_predicted):.4f}")
st.write(f"Root Mean Squared Error (RMSE) = {np.sqrt(metrics.mean_squared_error(y_televote_test, y_televote_predicted)):.4f}")

# ### Best Fit Regression Line for Stacked Model: Jury
st.write("### Jury Model: Best Fit Line")
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(y_jury_test, y_jury_predicted, color='magenta')
ax.set_title('Jury Points: Actual vs Predicted')
ax.set_xlabel('Actual Jury Points')
ax.set_ylabel('Predicted Jury Points')

# Line of Perfect Prediction
ax.plot([min(y_jury_test), max(y_jury_test)], [min(y_jury_test), max(y_jury_test)], color='red', linestyle='--', label='Line of Perfect Prediction')

# Plot Regression Line
reg_j = LinearRegression()
reg_j.fit(y_jury_test.values.reshape(-1, 1), y_jury_predicted)
jury_best_fit = reg_j.predict(y_jury_test.values.reshape(-1, 1))
ax.plot(y_jury_test, jury_best_fit, color='green', label='Regression Line (Best Fit)')
ax.legend()
st.pyplot(fig)

# ### Best Fit Regression Line for Stacked Model: Televote
st.write("### Televote Model: Best Fit Line")
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(y_televote_test, y_televote_predicted, color='orange')
ax.set_title('Televote Points: Actual vs Predicted')
ax.set_xlabel('Actual Televote Points')
ax.set_ylabel('Predicted Televote Points')

# Line of Perfect Prediction
ax.plot([min(y_televote_test), max(y_televote_test)], [min(y_televote_test), max(y_televote_test)], color='red', linestyle='--', label='Line of Perfect Prediction')

# Plot Regression Line
reg_t = LinearRegression()
reg_t.fit(y_televote_test.values.reshape(-1, 1), y_televote_predicted)
televote_best_fit = reg_t.predict(y_televote_test.values.reshape(-1, 1))
ax.plot(y_televote_test, televote_best_fit, color='green', label='Regression Line (Best Fit)')
ax.legend()
st.pyplot(fig)

# ### Conclusion of the Research
st.write("# Conclusion")
st.write(f"""
- **Jury Model**: Stacking Regressor R² value of **{r2_stack_jury:.4f}** shows a higher correlation with performance order compared to linear regression, where the R² was **{r2_jury:.4f}**.
- **Televote Model**: Stacking Regressor R² value of **{r2_stack_televote:.4f}** shows a stronger relationship with performance order, significantly higher than the simple linear regression model's R² of **{r2_televote:.4f}**.
""")

st.write("""
#### Observations:
- **Televote points** are more strongly influenced by the performance order, with a higher R² value indicating a stronger correlation.
- **Jury points** also show some dependency on performance order, but to a lesser degree.

While the **final_place** column significantly improved the model's R² score, it was not without its caveats, as it could potentially influence the result by essentially predicting the final ranking. Nonetheless, it was included to ensure that the model performed optimally, as omitting it led to negative R² values.

This analysis clearly demonstrates that **performance order** has a stronger impact on televote points than jury points in the context of the Eurovision Song Contest.
""")
