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
from src.util import save_dictionary, load_dictionary, serialize_dictionary

class TrainingStatistics():
    def __init__(self, single_lambda=False):
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
        self.single_lambda = single_lambda

    def log_statistics(self, fairness_rate, utilities, benefit_deltas, verbose=False):
        self.mean_utilities.append(utilities.mean(axis=0))
        self.stddev_utilites.append(utilities.std(axis=0))
        self.first_quartile_utilities.append(np.percentile(utilities, 25, axis=0))
        self.median_utilites.append(np.median(utilities, axis=0))
        self.third_quartile_utilities.append(np.percentile(utilities, 75, axis=0))
        self.mean_benefit_delta.append(benefit_deltas.mean(axis=0))
        self.stddev_benefit_delta.append(benefit_deltas.std(axis=0))
        self.first_quartile_benefit_delta.append(np.percentile(benefit_deltas, 25, axis=0))
        self.median_benefit_delta.append(np.median(benefit_deltas, axis=0))
        self.third_quartile_benefit_delta.append(np.percentile(benefit_deltas, 75, axis=0))
        self.fairness_rates.append(fairness_rate)

        if verbose:
            print("------------------- Utility ----------------------")
            print("Mean: {}".format(utilities.mean(axis=0)))
            print("Standard deviation: {}".format(utilities.std(axis=0)))
            print("First quartile: {}".format(np.percentile(utilities, 25, axis=0)))
            print("Median: {}".format(np.median(utilities, axis=0)))
            print("Last quartile: {}".format(np.percentile(utilities, 75, axis=0)))

            print("------------------- Benefit Delta ----------------")
            print("Mean: {}".format(benefit_deltas.mean(axis=0)))
            print("Standard deviation: {}".format(benefit_deltas.std(axis=0)))
            print("First quartile: {}".format(np.percentile(benefit_deltas, 25, axis=0)))
            print("Median: {}".format(np.median(benefit_deltas, axis=0)))
            print("Last quartile: {}".format(np.percentile(benefit_deltas, 75, axis=0)))
    
    def to_dictionary(self):
        if not self.single_lambda:
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
        else:
            return {
                "lambdas": self.fairness_rates,
                "utility": {
                    "mean": self.mean_utilities[0],
                    "stddev": self.stddev_utilites[0],
                    "first_quartile": self.first_quartile_utilities[0],
                    "median": self.median_utilites[0],
                    "third_quartile": self.third_quartile_utilities[0],
                },
                "benefit_delta": {
                    "mean": self.mean_benefit_delta[0],
                    "stddev": self.stddev_benefit_delta[0],
                    "first_quartile": self.first_quartile_benefit_delta[0],
                    "median": self.median_benefit_delta[0],
                    "third_quartile": self.third_quartile_benefit_delta[0],
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

def _training_iteration(training_parameters, store_all, verbose):
    utilities = []
    benefit_deltas = []
    policy_thetas = []
    i = 0 
    np.random.seed()
    for utility, benefit_delta, policy in consequential_learning(**training_parameters):
        if verbose:
            print("Timestep {}: \t Utility: {} \n\t Benefit Delta: {}".format(i, utility, benefit_delta))
        utilities.append(utility)
        benefit_deltas.append(benefit_delta)
        policy_thetas.append(policy.theta.copy().tolist())
        i += 1

    if store_all:
        return utilities, benefit_deltas, policy_thetas
    else:
        return utilities[-1], benefit_deltas[-1], policy_thetas[-1]

def _train_over_iterations(training_parameters, iterations, store_all, verbose, asynchronous):
    utilities_over_iterations = []
    benefit_deltas_over_iterations = []
    thetas_over_iterations = []

    # multithreaded runs of training
    if asynchronous:
        apply_results = []
        pool = mp.Pool(mp.cpu_count())
        for _ in range(0, iterations):
            apply_results.append(pool.apply_async(_training_iteration, args=(training_parameters, store_all, False)))
        pool.close()
        pool.join()

        for result in apply_results:
            utilities, benefit_deltas, thetas = result.get()
            utilities_over_iterations.append(utilities)
            benefit_deltas_over_iterations.append(benefit_deltas)
            thetas_over_iterations.append(thetas)
    else:
        for _ in range(0, iterations):
            utilities, benefit_deltas, thetas = _training_iteration(training_parameters, store_all, verbose)
            utilities_over_iterations.append(utilities)
            benefit_deltas_over_iterations.append(benefit_deltas)
            thetas_over_iterations.append(thetas)

    return np.array(utilities_over_iterations).squeeze(), np.array(benefit_deltas_over_iterations).squeeze(), thetas_over_iterations

def train(training_parameters, fairness_rates, iterations=30, verbose=False, asynchronous=True):
    """ Executes multiple runs of consequential learning with the same training parameters
    but different seeds for the specified fairness rates. 
        
    Args:
        training_parameters: The parameters used to configure the consequential learning algorithm.
        fairness_rates: An iterable containing all fairness rates for which consequential learning
        should be run and statistics will be collected.
        iterations: The number of times consequential learning will be run for one of the specified
        fairness rates. The resulting statistics will be applied over the number of runs
        verbose: A flag indicating if the results of each fairness rate should be printed.
        asynchronous: A flag indicating if the iterations should be executed asynchronously.

    Returns:
        training_statistic: A dictionary that contains statistical data about 
        the executed runs.
    """
    single_lambda = len(fairness_rates)==1
    statistics = TrainingStatistics(single_lambda=single_lambda)
    current_training_parameters = copy.deepcopy(training_parameters)

    if "save_path" in training_parameters:
        base_save_path = "{}/runs".format(training_parameters["save_path"])
        Path(base_save_path).mkdir(parents=True, exist_ok=True)
        runs = os.listdir(base_save_path)

        if len(runs) == 0:
            current_run = 0
        else:
            runs.sort()
            current_run = int(runs[-1].replace("run", "")) + 1

        base_save_path = "{}/{}".format(base_save_path, current_run)
        Path(base_save_path).mkdir(parents=True, exist_ok=True)
        parameter_save_path = "{}/parameters.json".format(base_save_path)

        serialized_dictionary = serialize_dictionary(training_parameters)
        serialized_dictionary["lambdas"] = fairness_rates
        save_dictionary(serialized_dictionary, parameter_save_path)
    else:
        base_save_path = None

    if training_parameters["data"]["keep_data_across_lambdas"]:
        current_training_parameters["data"] = _generate_data_set(training_parameters)

        if base_save_path is not None:
            data_save_path = "{}/data.json".format(base_save_path)
            data_dict = {
                "x": current_training_parameters["data"]["test_dataset"][0].tolist(),
                "s": current_training_parameters["data"]["test_dataset"][1].tolist(),
                "y": current_training_parameters["data"]["test_dataset"][2].tolist()
            }
            save_dictionary(data_dict, data_save_path)

    for fairness_rate in fairness_rates:
        print("--------------------------------------------------")
        print("------------------- Lambda {} -------------------".format(fairness_rate))
        print("--------------------------------------------------")
        current_training_parameters["optimization"]["fairness_rate"] = fairness_rate

        utilities, benefit_deltas, thetas = _train_over_iterations(current_training_parameters, iterations, single_lambda, verbose, asynchronous)

        statistics.log_statistics(fairness_rate, utilities, benefit_deltas, verbose)

        if base_save_path is not None:
            lambda_path = "{}/lambda{}/".format(base_save_path, fairness_rate)
            Path(lambda_path).mkdir(parents=True, exist_ok=True)
            model_save_path = "{}models.json".format(lambda_path)

            theta_dict = {str(i):theta for i, theta in enumerate(thetas)}

            save_dictionary(serialize_dictionary(theta_dict), model_save_path)

    return statistics.to_dictionary(), base_save_path