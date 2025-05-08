import pandas as pd

def load_excel(file_path, na_values=['NA'], sheet_name=0, skiprows=1):
    return pd.read_excel(file_path, sheet_name=sheet_name, index_col=None, na_values=na_values, skiprows=skiprows)

def load_csv(file_path, na_values=['NA'], skiprows=1, encoding='utf-8'):
    return pd.read_csv(file_path, index_col=None, na_values=na_values, skiprows=skiprows, encoding=encoding)

""" Example: merged_df = merge_dataframes([df_red, df_white], ['red', 'white'], label_column='type') this merges two dataframes and creates a label column"""
def merge_dataframes(dfs, labels, label_column='source'):
    if len(dfs) != len(labels):
        raise ValueError("The number of DataFrames and labels must be the same.")
    for df, label in zip(dfs, labels):
        df[label_column] = label
    return pd.concat(dfs, ignore_index=True)

def remove_missing(df):
    return df.dropna()

def remove_duplicates(df):
    return df.drop_duplicates()
