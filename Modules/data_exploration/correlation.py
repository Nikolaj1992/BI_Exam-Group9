import numpy as np
import pandas as pd


def pearson_corr(x, y):
    return np.corrcoef(x, y)[0, 1]

def correlation_matrix(dataframe):
    return dataframe.corr()

