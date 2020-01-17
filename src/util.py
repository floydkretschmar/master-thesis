import numpy as np
import json
np.seterr(divide='ignore', invalid='ignore', over='ignore')

def sigmoid(x):
    return np.where(x >= 0, 
                    1 / (1 + np.exp(-x)), 
                    np.exp(x) / (1 + np.exp(x)))

def save_dictionary(dictionary, path):
    try:
        with open(path, 'w') as file_path:
            json.dump(dictionary, file_path)
    except Exception as e:
        print('Saving file {} failed with exception: \n {}'.format(path, str(e)))

def load_dictionary(path):
    try:
        with open(path, 'r') as file_path:
            return json.load(file_path)
    except Exception as e:
        print('Loading file {} failed with exception: \n {}'.format(path, str(e)))
        return None