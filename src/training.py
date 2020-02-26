import os
import sys
root_path = os.path.abspath(os.path.join('.'))
if root_path not in sys.path:
    sys.path.append(root_path)

import numpy as np
import multiprocessing as mp
import time
from pathlib import Path
from copy import deepcopy
import numbers

from src.consequential_learning import ConsequentialLearning, DualGradientConsequentialLearning
from src.util import save_dictionary, load_dictionary, serialize_dictionary, stack, check_for_missing_kwargs
from src.training_evaluation import Statistics, MultiStatistics, ModelParameters

class _Trainer():
    def _training_iteration(self, training_parameters, training_method):
        np.random.seed()
        results_over_lambdas = []

        for decisions_over_time, model_parameters in training_method(training_parameters):
            statistics = Statistics(
                    predictions=decisions_over_time, 
                    observations=training_parameters["data"]["test"][0],
                    protected_attributes=training_parameters["data"]["test"][1], 
                    ground_truths=training_parameters["data"]["test"][2], 
                    utility_function=training_parameters["model"]["utility_function"])
            results_over_lambdas.append((statistics, deepcopy(model_parameters)))

        return results_over_lambdas

    @staticmethod
    def _process_results(results_per_iterations):
        statistics_over_lambdas = []
        model_parameters_over_lambdas = {}

        for num_iteration, iteration_result in enumerate(results_per_iterations):
            for num_lambda, lambda_result in enumerate(iteration_result):
                statistics, model_parameters = lambda_result

                if num_lambda >= len(statistics_over_lambdas):
                    statistics_over_lambdas.append(statistics)
                else:
                    statistics_over_lambdas[num_lambda].merge(statistics)
                
                if num_lambda in model_parameters_over_lambdas:
                    model_parameters_over_lambdas[num_lambda][num_iteration] = model_parameters
                else:
                    model_parameters_over_lambdas[num_lambda] = {
                        num_iteration: model_parameters
                    }
                    
        return statistics_over_lambdas, model_parameters_over_lambdas

    def train_over_iterations(self, training_parameters, training_method, iterations, asynchronous):        
        results_per_iterations = []

        # multithreaded runs of training
        if asynchronous:
            apply_results = []
            results_per_iterations = []
            pool = mp.Pool(mp.cpu_count())
            for _ in range(0, iterations):
                apply_results.append(pool.apply_async(self._training_iteration, args=(training_parameters, training_method)))
            pool.close()
            pool.join()
            
            for result in apply_results:
                results_per_iterations.append(result.get())
        else:
            results_per_iterations = []
            for _ in range(0, iterations):
                results_per_iterations.append(self._training_iteration(training_parameters, training_method))
            
        total_statistics, total_parameters = self._process_results(results_per_iterations)
        if len(total_statistics) == 1:
            return total_statistics[0], total_parameters

        return total_statistics, total_parameters


def _check_for_missing_training_parameters(training_parameters):
    """ Checks the dictionary of training parameters specified by the user for missing entries.
        
    Args:
        training_parameters: The parameters used to configure the consequential learning algorithm.
    """
    check_for_missing_kwargs("training()", ["experiment_name", "model", "parameter_optimization", "data"], training_parameters)
    check_for_missing_kwargs(
        "training()", 
        ["benefit_function", "utility_function", "fairness_function", "fairness_gradient_function", "feature_map", "learn_on_entire_history", "use_sensitve_attributes", "bias", "initial_theta", "initial_lambda"], 
        training_parameters["model"])
    check_for_missing_kwargs("training()", ["distribution", "fraction_protected", "num_test_samples"], training_parameters["data"])
    check_for_missing_kwargs("training()", ["time_steps", "epochs", "batch_size", "learning_rate", "decay_rate", "decay_step", "num_decisions"], training_parameters["parameter_optimization"])

    if "lagrangian_optimization" in training_parameters:
        check_for_missing_kwargs("training()", ["epochs", "batch_size", "learning_rate", "decay_rate", "decay_step", "num_decisions"], training_parameters["lagrangian_optimization"])


def _generate_data_set(training_parameters):
    """ Generates one training and test dataset to be used across all lambdas.
        
    Args:
        training_parameters: The parameters used to configure the consequential learning algorithm.

    Returns:
        data: A dictionary containing both the test and training dataset.
    """
    num_decisions = training_parameters["parameter_optimization"]["num_decisions"]
    distribution = training_parameters["data"]["distribution"]
    fraction_protected = training_parameters["data"]["fraction_protected"]

    test_dataset = distribution.sample_dataset(
        n=training_parameters["data"]["num_test_samples"], 
        fraction_protected=fraction_protected)
    theta_train_x, theta_train_s, theta_train_y = distribution.sample_dataset(
        n=num_decisions * training_parameters["parameter_optimization"]["time_steps"], 
        fraction_protected=fraction_protected)
    

    theta_train_datasets = []
    for i in range(0, theta_train_x.shape[0], num_decisions):
        theta_train_datasets.append((theta_train_x[i:i+num_decisions], theta_train_s[i:i+num_decisions], theta_train_y[i:i+num_decisions]))
    
    data = {
        'training': {
            "theta": theta_train_datasets
        },
        'test': test_dataset
    } 

    if "lagrangian_optimization" in training_parameters:
        num_decisions_lambda = training_parameters["lagrangian_optimization"]["num_decisions"]
        data["training"]["lambda"] = distribution.sample_dataset(
            n=num_decisions_lambda, 
            fraction_protected=fraction_protected)

    return data


def _prepare_training(training_parameters):
    """ Preprocesses the training parameters and sets up the training procedure
        
    Args:
        training_parameters: The parameters used to configure the consequential learning algorithm.

    Returns:
        current_training_parameters: The pre processed training parameters.
    """
    _check_for_missing_training_parameters(training_parameters)
    current_training_parameters = deepcopy(training_parameters)

    if "save_path" in training_parameters:
        base_save_path = "{}/res/{}".format(training_parameters["save_path"], training_parameters["experiment_name"])
        Path(base_save_path).mkdir(parents=True, exist_ok=True)

        timestamp = time.gmtime()
        ts_folder = time.strftime("%Y-%m-%d-%H-%M-%S", timestamp)
        base_save_path = "{}/{}".format(base_save_path, ts_folder)
        Path(base_save_path).mkdir(parents=True, exist_ok=True)

        parameter_save_path = "{}/parameters.json".format(base_save_path)

        serialized_dictionary = serialize_dictionary(current_training_parameters)
        save_dictionary(serialized_dictionary, parameter_save_path)
    else:
        base_save_path = None

    current_training_parameters["data"] = _generate_data_set(training_parameters)

    if base_save_path is not None:
        data_save_path = "{}/data.json".format(base_save_path)
        data_dict = {
            "x": current_training_parameters["data"]["test"][0].tolist(),
            "s": current_training_parameters["data"]["test"][1].tolist(),
            "y": current_training_parameters["data"]["test"][2].tolist()
        }
        save_dictionary(data_dict, data_save_path)
        del data_dict

    return current_training_parameters, base_save_path
     

def _save_results(base_save_path, statistics, model_parameters=None, sub_directory=None):
    """ Stores the training results (statistics and/or model parameters) in the specified path.
        
    Args:
        base_save_path: The base path specified by the user under which the results will be stored.
        statistics: The statistics of either a specific training run for a specified lambda, or the overall statistics over all lambdas.
        model_parameters: The parameters of a model for a specific lambda.
        sub_directory: The subdirectory under which the results will be stored.
    """
    # save the model parameters to be able to restore the model
    if sub_directory is not None:
        lambda_path = "{}/{}/".format(base_save_path, sub_directory)
        Path(lambda_path).mkdir(parents=True, exist_ok=True)
    else:
        lambda_path = "{}/".format(base_save_path)

    if model_parameters is not None:
        model_save_path = "{}models.json".format(lambda_path)

        serialized_model_parameters = serialize_dictionary(model_parameters)
        save_dictionary(serialized_model_parameters, model_save_path)

    # save the results for each lambda
    statistics_save_path = "{}statistics.json".format(lambda_path)
    serialized_statistics = serialize_dictionary(statistics.to_dict())
    save_dictionary(serialized_statistics, statistics_save_path)


def train(training_parameters, iterations=30, asynchronous=True):
    """ Executes multiple runs of consequential learning with the same training parameters
    but different seeds for the specified fairness rates. 
        
    Args:
        training_parameters: The parameters used to configure the consequential learning algorithm.
        iterations: The number of times consequential learning will be run for one of the specified
        asynchronous: A flag indicating if the iterations should be executed asynchronously.

    Returns:
        training_statistic: A dictionary that contains statistical data about 
        the executed runs.
        model_parameters: A single ModelParameters object if training both lambda and theta,
        or None, when training with fixed lambdas. The ModelParameter object contains theta 
        and lambda values for each timestep over all iterations.
    """
    assert iterations > 0

    current_training_parameters, base_save_path = _prepare_training(training_parameters)
    trainer = _Trainer()

    if isinstance(current_training_parameters["model"]["initial_lambda"], numbers.Number) and "lagrangian_optimization" not in training_parameters:
        print("---------- Single training run for fixed lambda ----------")
        training_algorithm = ConsequentialLearning(current_training_parameters["model"]["learn_on_entire_history"])
        overall_statistics, model_parameters = trainer.train_over_iterations(current_training_parameters, training_algorithm.train, iterations, asynchronous)
    elif not isinstance(current_training_parameters["model"]["initial_lambda"], numbers.Number):
        print("---------- Training with fixed lambdas ----------")
        fairness_rates = deepcopy(current_training_parameters["model"]["initial_lambda"])
        training_algorithm = ConsequentialLearning(current_training_parameters["model"]["learn_on_entire_history"])
        overall_statistics = MultiStatistics("log", fairness_rates, "Lambda")

        for fairness_rate in fairness_rates:
            current_training_parameters["model"]["initial_lambda"] = fairness_rate
            statistics, model_parameters = trainer.train_over_iterations(current_training_parameters, training_algorithm.train, iterations, asynchronous)

            overall_statistics.log_run(statistics)
            if base_save_path is not None:  
                _save_results(
                    base_save_path=base_save_path, 
                    statistics=statistics, 
                    model_parameters=model_parameters, 
                    sub_directory="lambda_{}".format(fairness_rate))

    elif "lagrangian_optimization" in training_parameters:
        print("---------- Training both theta and lambda ----------")
        training_algorithm = DualGradientConsequentialLearning(current_training_parameters["model"]["learn_on_entire_history"])
        
        lamda_iterations = range(0, current_training_parameters["lagrangian_optimization"]["iterations"])
        sub_directories = ["lambda_iteration_{}".format(lamb) for lamb in lamda_iterations]
        overall_statistics = MultiStatistics("linear", lamda_iterations, "Lambda Training Iteration")
        
        statistics, model_parameters = trainer.train_over_iterations(current_training_parameters, training_algorithm.train, iterations, asynchronous)

        for num_lambda, statistic in enumerate(statistics):
            overall_statistics.log_run(statistic)

            if base_save_path is not None:  
                _save_results(
                    base_save_path=base_save_path, 
                    statistics=statistic, 
                    model_parameters=model_parameters[num_lambda], 
                    sub_directory=sub_directories[num_lambda])
            
    # save the overall results    
    if base_save_path is not None:
        _save_results(
            base_save_path=base_save_path, 
            statistics=overall_statistics)

    return overall_statistics, ModelParameters(model_parameters), base_save_path
    
