import numpy as np
from typing import List

__all__ = ['variance', 'standard_deviation', 'range_stat', 'iqr', 'remove_outliers']

def variance(data):
    return np.var(data, ddof=1)

def standard_deviation(data):
    return np.std(data, ddof=1)

def range_stat(data):
    return np.max(data) - np.min(data)

def iqr(data):
    q3, q1 = np.percentile(data, [75, 25])
    return q3 - q1

''' Example: remove_outliers(df, df['length']) '''
def remove_outliers(df, data: List):
    q1 = data.quantile(.25)
    q3 = data.quantile(.75)
    IQR = q3 - q1
    # values smaller than 1.5 IQR below q1 and bigger that 1.5 IQR over q3 
    outliers = df[(data < (q1 - 1.5 * IQR)) | (data > (q3 + 1.5 * IQR))]
    df = df.drop(outliers.index, inplace=True)
    return df
    