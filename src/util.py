import numpy as np
import json
import numbers
import inspect
import copy
np.seterr(divide='ignore', invalid='ignore', over='ignore')

def sigmoid(x):
    return np.where(x >= 0, 
                    1 / (1 + np.exp(-x)), 
                    np.exp(x) / (1 + np.exp(x)))

def save_dictionary(dictionary, path):
    try:
        with open(path, 'w+') as file_path:
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

def serialize_dictionary(dictionary):
    serialized_dict = copy.deepcopy(dictionary)
    for key, value in serialized_dict.items():
        if isinstance(value, dict):
            serialized_dict[key] = serialize_dictionary(value)
        elif isinstance(value, np.ndarray):
            serialized_dict[key] = value.tolist()
        elif inspect.isfunction(value):
            serialized_dict[key] = value.__name__
        elif not (isinstance(value, str) or isinstance(value, numbers.Number) or isinstance(value, list) or isinstance(value, bool)):
            serialized_dict[key] = type(value).__name__

    return serialized_dict

def check_for_missing_kwargs(function_name, required_kwargs, kwargs):
    missing_kwargs = []
    for required_kwarg in required_kwargs:
        if required_kwarg not in kwargs:
            missing_kwargs.append(required_kwarg)

    num_missing_kwargs = len(missing_kwargs)
    if num_missing_kwargs > 0:
        raise TypeError("{} missing {} required positional {}: {}".format(function_name, num_missing_kwargs, "argument" if num_missing_kwargs == 1 else "arguments", ", ".join(missing_kwargs)))