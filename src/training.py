import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
import multiprocessing as mp
import copy
from pathlib import Path

from src.consequential_learning import consequential_learning
from src.util import save_dictionary, load_dictionary

class TrainingStatistics():
    def __init__(self):
        self.mean_utilities = []
        self.stddev_utilites = []
        self.first_quartile_utilities = []
        self.median_utilites = []
        self.third_quartile_utilities = []
        self.mean_benefit_delta = []
        self.stddev_benefit_delta = []
        self.first_quartile_benefit_delta = []
        self.median_benefit_delta = []
        self.third_quartile_benefit_delta = []
        self.fairness_rates = []

    def log_statistics(self, fairness_rate, utilities, benefit_deltas, verbose=False):
        self.mean_utilities.append(utilities.mean())
        self.stddev_utilites.append(utilities.std())
        self.first_quartile_utilities.append(np.percentile(utilities, 25))
        self.median_utilites.append(np.median(utilities))
        self.third_quartile_utilities.append(np.percentile(utilities, 75))
        self.mean_benefit_delta.append(benefit_deltas.mean())
        self.stddev_benefit_delta.append(benefit_deltas.std())
        self.first_quartile_benefit_delta.append(np.percentile(benefit_deltas, 25))
        self.median_benefit_delta.append(np.median(benefit_deltas))
        self.third_quartile_benefit_delta.append(np.percentile(benefit_deltas, 75))
        self.fairness_rates.append(fairness_rate)

        if verbose:
            print("------------------- Utility ----------------------")
            print("Mean: {}".format(utilities.mean()))
            print("Standard deviation: {}".format(utilities.std()))
            print("First quartile: {}".format(np.percentile(utilities, 25)))
            print("Median: {}".format(np.median(utilities)))
            print("Last quartile: {}".format(np.percentile(utilities, 75)))

            print("------------------- Benefit Delta ----------------")
            print("Mean: {}".format(benefit_deltas.mean()))
            print("Standard deviation: {}".format(benefit_deltas.std()))
            print("First quartile: {}".format(np.percentile(benefit_deltas, 25)))
            print("Median: {}".format(np.median(benefit_deltas)))
            print("Last quartile: {}".format(np.percentile(benefit_deltas, 75)))
    
    def to_json(self):
        return {
            "lambdas": self.fairness_rates,
            "utility": {
                "mean": np.array(self.mean_utilities),
                "stddev": np.array(self.stddev_utilites),
                "first_quartile": np.array(self.first_quartile_utilities),
                "median": np.array(self.median_utilites),
                "third_quartile": np.array(self.third_quartile_utilities),
            },
            "benefit_delta": {
                "mean": np.array(self.mean_benefit_delta),
                "stddev": np.array(self.stddev_benefit_delta),
                "first_quartile": np.array(self.first_quartile_benefit_delta),
                "median": np.array(self.median_benefit_delta),
                "third_quartile": np.array(self.third_quartile_benefit_delta),
            }
        }

def _generate_data_set(training_parameters):
    """ Generates one training and test dataset to be used across all lambdas.
        
    Args:
        training_parameters: The parameters used to configure the consequential learning algorithm.

    Returns:
        data: A dictionary containing both the test and training dataset.
    """
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

result_list = []
def _result_worker(result):
    global result_list
    result_list.append(result)

def _train_single(training_parameters, model_save_path=None, verbose=False):
    """ Executes multiple runs of consequential learning with the same training parameters
    but different seeds.
        
    Args:
        training_parameters: The parameters used to configure the consequential learning algorithm.
        iterations: The number of times consequential learning will be run.

    Returns:
        utility: The utility of the trained policy on the last time step.
        benefit_delta: The benefit delta of the trained policy on the last time step.
    """
    i = 0 
    np.random.seed()
    for u, bd, pi in consequential_learning(**training_parameters):
        if verbose:
            print("Timestep {}: \t Utility: {} \n\t Benefit Delta: {}".format(i, u, bd))
        utility, benefit_delta, policy = u, bd, pi
        i += 1

    if "save_path" in training_parameters["model"]:
        save_dictionary({"theta": policy.theta.tolist()}, model_save_path)

    return utility, benefit_delta

def train_multiple(training_parameters, iterations, verbose=False, asynchronous=True):
    """ Executes multiple runs of consequential learning with the same training parameters
    but different seeds.
        
    Args:
        training_parameters: The parameters used to configure the consequential learning algorithm.
        iterations: The number of times consequential learning will be run.

    Returns:
        training_statistic: A TrainingStatistics object that contains statistical data about 
        the executed runs.
    """

    statistics = TrainingStatistics()
    current_train_parameters = copy.deepcopy(training_parameters)

    if "save_path" in training_parameters["model"]:
        base_save_path = training_parameters["model"]["save_path"]
        runs = os.listdir(base_save_path)

        if len(runs) == 0:
            current_run = 0
        else:
            runs.sort()
            current_run = int(runs[-1].replace("run", "")) + 1

        base_save_path = "{}/run{}/".format(base_save_path, current_run)
        Path(base_save_path).mkdir(parents=True, exist_ok=True)
    else:
        model_save_directory = None

    if current_train_parameters["data"]["keep_data_across_lambdas"]:
        current_train_parameters["data"] = _generate_data_set(training_parameters)

    for fairness_rate in current_train_parameters["optimization"]["fairness_rates"]:
        global result_list

        if base_save_path is not None:
            model_save_directory = "{}/lambda{}/".format(base_save_path, fairness_rate)
            Path(model_save_directory).mkdir(parents=True, exist_ok=True)

        print("--------------------------------------------------")
        print("------------------- Lambda {} -------------------".format(fairness_rate))
        print("--------------------------------------------------")
        benefit_deltas = []
        utilities = []
        current_train_parameters["optimization"]["fairness_rate"] = fairness_rate

        # multithreaded runs of training
        if asynchronous:
            pool = mp.Pool(mp.cpu_count())
            for j in range(0, iterations):
                if model_save_directory is not None:
                    model_save_path = "{}/model_{}.json".format(model_save_directory, j)

                pool.apply_async(_train_single, args=(current_train_parameters, model_save_path), callback=_result_worker) 
            pool.close()
            pool.join()
        else:
            for j in range(0, iterations):
                if model_save_directory is not None:
                    model_save_path = "{}/model_{}.json".format(model_save_directory, j)

                result = _train_single(current_train_parameters, model_save_path, verbose)
                _result_worker(result)

        results = np.array(result_list).squeeze()
        result_list = []

        utilities = results[:,0]
        benefit_deltas = results[:,1]

        statistics.log_statistics(fairness_rate, utilities, benefit_deltas, verbose)

    return statistics.to_json()