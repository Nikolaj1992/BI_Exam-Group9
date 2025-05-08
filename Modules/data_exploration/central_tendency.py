import numpy as np

def mean(data):
    return np.mean(data)

def median(data):
    return np.median(data)

def mode(data):
    from scipy import stats
    mode_res = stats.mode(data, keepdims=True)
    return mode_res.mode[0], mode_res.count[0]