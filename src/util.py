import copy
import inspect
import json
import numbers

import numpy as np
import torch
from numpy.random import RandomState

np.seterr(divide='ignore', invalid='ignore', over='ignore')

def to_device(torch_object):
    if torch.cuda.is_available():
        return torch_object.cuda()
    else:
        return torch_object

def from_device(torch_object):
    if torch_object.is_cuda:
        return torch_object.cpu()
    else:
        return torch_object

def get_random(seed=None):
    if seed is None:
        return RandomState()
    else:
        return RandomState(seed)

def train_validation_split(data, validation_percentage=0.2):
    data_len = len(data)
    validation_len = int(data_len*validation_percentage)

    if torch.is_tensor(data):
        data_train, data_val = torch.utils.data.random_split(data, [data_len-validation_len, validation_len])
        return data[data_train.indices], data[data_val.indices]
    elif isinstance(data, np.ndarray):
        return train_test_split(data, test_percentage=validation_percentage)


def get_list_of_seeds(number_of_seeds):
    max_value = 2 ** 32 - 1
    seeds = get_random().randint(
        0,
        max_value,
        size=number_of_seeds,
        dtype=np.dtype("int64"))
    return seeds

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
        raise TypeError("{} missing {} required positional {}: {}".format(function_name, num_missing_kwargs,
                                                                          "argument" if num_missing_kwargs == 1 else "arguments",
                                                                          ", ".join(missing_kwargs)))


def mean(target, axis=None):
    if isinstance(target, np.ndarray) or isinstance(target, numbers.Number):
        return np.mean(target, axis=axis)
    elif torch.is_tensor(target):
        return torch.mean(target, dim=axis)
    else:
        raise TypeError("The given target {} is neither a ndarray nor a torch tensor.".format(target))

def mean_difference(target, group_indicator, groups=[0, 1]):
    """ Calculates the mean difference of the target with regards to the sensitive attribute.

    Args:
        target: The target for which the mean difference will be calculated.
        group_indicator: The indicator vector that defines the group assignments of the target.
        gtoup: The values defining the groups according to the group_indicator vector

    Returns:
        mean_difference: The mean difference of the target
    """
    assert target.shape[0] == group_indicator.shape[0]

    if len(group_indicator) < 2:
        return 0.0

    s_idx = np.expand_dims(np.arange(group_indicator.shape[0]), axis=1)
    s_0_mask = group_indicator == groups[0]
    s_1_mask = group_indicator == groups[1]

    # print(s_1_mask.shape)
    # print(s_idx.shape)
    s_0_idx = s_idx[s_0_mask]
    s_1_idx = s_idx[s_1_mask]

    if len(s_0_idx) == 0 or len(s_1_idx) == 0:
        return 0.0

    target_s0 = mean(target[s_0_idx], axis=0)
    target_s1 = mean(target[s_1_idx], axis=0)

    return target_s0 - target_s1

def sort(object):
    if torch.is_tensor(object):
        sorted, _ = torch.sort(object)
        return sorted
    else:
        return np.sort(object)

def stack(base_array, new_array, axis):
    if base_array is None:
        return new_array
    elif isinstance(base_array, np.ndarray) and isinstance(new_array, np.ndarray):
        if axis == 0:
            return np.vstack((base_array, new_array))
        elif axis == 1:
            return np.hstack((base_array, new_array))
        else:
            return np.dstack((base_array, new_array))
    elif torch.is_tensor(base_array) and torch.is_tensor(new_array):
        return torch.cat((base_array, new_array), dim=axis)
    else:
        raise TypeError("Objects are either incompatible or neither a ndarray nor a torch tensor.")


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


def train_test_split(*arrays, test_percentage):
    assert len(arrays) > 0
    indices = np.array(range(arrays[0].shape[0]))

    boundary = int(len(indices) * test_percentage)
    test_indices, train_indices = np.split(get_random().permutation(indices), [boundary])

    splits = []
    for array in arrays:
        splits.extend((array[train_indices], array[test_indices]))

    return tuple(splits)