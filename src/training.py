import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
import multiprocessing as mp
import copy

from src.consequential_learning import consequential_learning
from src.util import save_dictionary, load_dictionary

result_list = []

def generate_data_set(training_parameters):
    num_decisions = training_parameters["data"]["num_decisions"]

    test_dataset = training_parameters["data"]["distribution"].sample_dataset(
        training_parameters["data"]["num_test_samples"], 
        training_parameters["data"]["fraction_protected"])
    train_x, train_s, train_y = training_parameters["data"]["distribution"].sample_dataset(
        num_decisions * training_parameters["optimization"]["time_steps"], 
        training_parameters["data"]["fraction_protected"])

    train_datasets = []
    for i in range(0, train_x.shape[0], num_decisions):
        train_datasets.append((train_x[i:i+num_decisions], train_s[i:i+num_decisions], train_y[i:i+num_decisions]))
        
    data = {
        'keep_data_across_lambdas': True,
        'training_datasets': train_datasets,
        'test_dataset': test_dataset
    }
        
    return data

def result_worker(result):
    global result_list
    result_list.append(result)

def train_single(training_parameters):
    np.random.seed()
    for u, bd, pi in consequential_learning(**training_parameters):
        utility, benefit_delta, policy = u, bd, pi

    if "save_path" in training_parameters["model"]:
        save_dictionary({"theta": policy.theta.tolist()}, training_parameters["model"]["save_path"])

    return utility, benefit_delta

def train_multiple(training_parameters, iterations, verbose=False, asynchronous=True):
    mean_utilities = []
    stddev_utilites = []
    mean_benefit_delta = []
    stddev_benefit_delta = []
    current_train_parameters = copy.deepcopy(training_parameters)
    base_model_save_path = training_parameters["model"]["save_path"]

    if current_train_parameters["data"]["keep_data_across_lambdas"]:
        current_train_parameters["data"] = generate_data_set(training_parameters)

    for fairness_rate in current_train_parameters["optimization"]["fairness_rates"]:
        global result_list

        print("Processing Lambda: {}".format(fairness_rate))
        benefit_deltas = []
        utilities = []
        current_train_parameters["optimization"]["fairness_rate"] = fairness_rate
        
        # multithreaded runs of training
        if asynchronous:
            pool = mp.Pool(mp.cpu_count())
            for j in range(0, iterations):
                model_save_path = "{}/lambda{}_model{}.json".format(base_model_save_path, fairness_rate, j)
                if model_save_path is not None:
                    current_train_parameters["model"]["save_path"] = model_save_path

                pool.apply_async(train_single, args=(current_train_parameters,), callback=result_worker) 
            pool.close()
            pool.join()
        else:
            for j in range(0, iterations):
                model_save_path = "{}/lambda{}_model{}.json".format(base_model_save_path, fairness_rate, j)
                if model_save_path is not None:
                    current_train_parameters["model"]["save_path"] = model_save_path

                result = train_single(current_train_parameters)
                result_worker(result)

        results = np.array(result_list).squeeze()
        result_list = []

        utilities = results[:,0]
        benefit_deltas = results[:,1]

        mean_utilities.append(utilities.mean())
        stddev_utilites.append(utilities.std())
        mean_benefit_delta.append(benefit_deltas.mean())
        stddev_benefit_delta.append(benefit_deltas.std())

        if verbose:
            print("Mean utility: {}".format(utilities.mean()))
            print("Stddev utility: {}".format(utilities.std()))
            print("Mean benefit delta: {}".format(benefit_deltas.mean()))
            print("Stddev benefit delta: {}".format(benefit_deltas.std()))

    return {
        "utility_stats": {
            "mean": np.array(mean_utilities),
            "stddev": np.array(stddev_utilites)
        },
        "benefit_delta_stats": {
            "mean": np.array(mean_benefit_delta),
            "stddev": np.array(stddev_benefit_delta)
        }
    }