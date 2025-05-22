import matplotlib.pyplot as plt
import seaborn as sbs
import numpy as np
import pandas as pd
import sys
import plotly.express as px
import plotly.graph_objects as go

def histogram(data, title='Histogram'):
    # Loop through numeric columns
    numeric_columns = data.select_dtypes(include='number').columns  # Get numeric columns
    for col in numeric_columns:
        plt.figure(figsize=(8, 6))  # Create a new figure with smaller size for each column
        plt.hist(data[col], bins=15, color='skyblue', edgecolor='black')  # Plot histogram
        plt.title(f"{title} - {col}")  # Title with column name
        plt.xlabel(col)  # Label x-axis with column name
        plt.ylabel('Frequency')  # Label y-axis
        
        plt.tight_layout()  # Adjust the layout to prevent overlap
        plt.show()  # Show the plot

def countplot_comparison(df_list, labels, category_col, hue_col=None, title_prefix='', figsize=(12, 4)):
    num_plots = len(df_list)
    plt.figure(figsize=figsize)
    for i, (df, label) in enumerate(zip(df_list, labels), 1):
        plt.subplot(1, num_plots, i)
        if hue_col and i == num_plots:
            sbs.countplot(x=category_col, hue=hue_col, data=df)
        else:
            sbs.countplot(x=category_col, data=df)
        plt.title(f'{title_prefix} {label}'.strip())
    plt.tight_layout()
    plt.show()

def boxplot(data=None, x=None, y=None, title='Boxplot', xlabel=None, ylabel=None, fig=None, ax=None):
    if not fig or not ax:
        fig, ax = plt.subplots(figsize=(6, 4))
    #plt.figure(figsize=(6, 4))
    sbs.boxplot(data=data, x=x, y=y, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel if xlabel else x)
    ax.set_ylabel(ylabel if ylabel else y)
    #plt.title(title)
    #plt.xlabel(xlabel if xlabel else x)
    #plt.ylabel(ylabel if ylabel else y)
    fig.tight_layout()
    if 'streamlit' not in sys.modules:
        plt.show()
        return
    return fig

def interactive_boxplot(data, x, y, title='Interactive Boxplot', xlabel=None, ylabel=None):
    fig = px.box(data, x=x, y=y, points="all", title=title,
                 labels={x: xlabel or x, y: ylabel or y})
    return fig

def barplot(df, x_col, y_col=None, agg_func='mean', title='Bar Plot', xlabel=None, ylabel=None, hue_col=None):
    # Handle aggregation if no y_col is provided
    if y_col is None:
        # Aggregate values by x_col
        if agg_func == 'mean':
            df_agg = df.groupby(x_col).mean().reset_index()
        elif agg_func == 'sum':
            df_agg = df.groupby(x_col).sum().reset_index()
        else:
            raise ValueError("Aggregation function not supported. Use 'mean' or 'sum'.")
        sbs.barplot(x=x_col, y=y_col, data=df_agg)
    else:
        sbs.barplot(x=x_col, y=y_col, data=df, hue=hue_col)
    plt.title(title)
    plt.xlabel(xlabel if xlabel else x_col.capitalize())
    plt.ylabel(ylabel if ylabel else y_col.capitalize())
    plt.show()

def tooltip_barplot(data, x_col, y_col, title='Barplot with Tooltips', xlabel=None, ylabel=None):
    fig = px.bar(
        data, x=x_col, y=y_col,
        hover_data={x_col: True, y_col: ':.2f'},
        title=title,
        labels={x_col: xlabel or x_col, y_col: ylabel or y_col}
    )
    return fig

def lmplot(df, x, y, hue=None, title='Linear Regression Plot', figsize=(6, 4)):
    plt.figure(figsize=figsize)
    sbs.lmplot(x=x, y=y, hue=hue, data=df)
    plt.title(title)
    plt.tight_layout()
    plt.show()

def scatter_plot(x, y, title='Scatter Plot'):
    plt.scatter(x, y)
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

def scatter_3d(data, x, y, z, color=None, size=None, title='3D Scatter Plot', xlabel=None, ylabel=None, zlabel=None):
    fig = px.scatter_3d(data, x=x, y=y, z=z, color=color, size=size, title=title)
    fig.update_layout(scene=dict(
        xaxis_title=xlabel or x,
        yaxis_title=ylabel or y,
        zaxis_title=zlabel or z
    ),margin=dict(
        l=0, 
        r=0, 
        b=0, 
        t=40
    ))
    return fig

def correlation_heatmap(dataframe, title='Correlation Heatmap'):
    sbs.heatmap(dataframe.corr(), annot=True, cmap='coolwarm')
    plt.title(title)
    plt.show()

def zscore_heatmap(zscores, title='Z-Score Heatmap'):
    z_df = pd.DataFrame(zscores).T if isinstance(zscores, dict) else zscores
    fig = px.imshow(z_df,
                    color_continuous_scale='RdBu_r',
                    text_auto=True,
                    labels=dict(color="Z-Score"),
                    title=title)
    return fig
