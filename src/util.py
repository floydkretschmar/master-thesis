import numpy as np
np.seterr(divide='ignore', invalid='ignore', over='ignore')

def sigmoid(x):
    return np.where(x >= 0, 
                    1 / (1 + np.exp(-x)), 
                    np.exp(x) / (1 + np.exp(x)))