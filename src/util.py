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

def serialize_value(value):
    if isinstance(value, dict):
        return serialize_dictionary(value)
    elif isinstance(value, list):
        return serialize_list(value)
    elif isinstance(value, np.ndarray):
        return value.tolist()
    elif inspect.isfunction(value):
        return value.__name__
    elif not (isinstance(value, str) or isinstance(value, numbers.Number) or isinstance(value, list) or isinstance(value, bool)):
        return type(value).__name__
    else:
        return value

def serialize_dictionary(dictionary):
    serialized_dict = copy.deepcopy(dictionary)
    for key, value in serialized_dict.items():
        serialized_dict[key] = serialize_value(value)

    return serialized_dict

def serialize_list(unserialized_list):
    serialized_list = []
    for value in unserialized_list:
        serialized_list.append(serialize_value(value))

    return serialized_list

def check_for_missing_kwargs(function_name, required_kwargs, kwargs):
    missing_kwargs = []
    for required_kwarg in required_kwargs:
        if required_kwarg not in kwargs:
            missing_kwargs.append(required_kwarg)

    num_missing_kwargs = len(missing_kwargs)
    if num_missing_kwargs > 0:
        raise TypeError("{} missing {} required positional {}: {}".format(function_name, num_missing_kwargs, "argument" if num_missing_kwargs == 1 else "arguments", ", ".join(missing_kwargs)))

def stack(base_array, new_array, axis):
    if base_array is None:
        return new_array
    else:
        if axis == 0:
            return np.vstack((base_array, new_array))
        elif axis == 1:
            return np.hstack((base_array, new_array))
        else:
            return np.dstack((base_array, new_array))


def load_dataset(data_file_path):
    """
    Load a dataset from the specified path.

    Args:
        data_file_path: The file path of the dataset.

    Returns:
        x: The NxD matrix of non-sensitive feature vectors.
        s: The Nx1 vector of sensitive features. If the dataset contained more than one sensitive
           feature, the first one is chosen.
        y: The Nx1 ground truth data.
    """
    raw_data = np.load(data_file_path)
    x = raw_data["x"]
    y = raw_data["y"]
    s = raw_data["s"]

    if s.shape[1] > 1:
        s = s[:, 0]

    x = whiten(x)
    return x, s.astype(int), y
    

def whiten(data, columns=None, conditioning=1e-8):
    """
    Whiten various datasets in data dictionary.

    Args:
        data: Data array.
        columns: The columns to whiten. If `None`, whiten all.
        conditioning: Added to the denominator to avoid divison by zero.
    """
    if columns is None:
        columns = np.arange(data.shape[1])
    mu = np.mean(data[:, columns], 0)
    std = np.std(data[:, columns], 0)
    data[:, columns] = (data[:, columns] - mu) / (std + conditioning)
    return data

def train_test_split(x, y, s, test_size):
    indices = np.array(range(x.shape[0]))

    boundary = int(len(indices) * test_size)
    test_indices, train_indices = np.split(np.random.permutation(indices), [boundary])
    return x[train_indices], x[test_indices], y[train_indices], y[test_indices], s[train_indices], s[test_indices]