import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
import multiprocessing as mp
from pathlib import Path
from copy import deepcopy

from src.consequential_learning import consequential_learning
from src.util import save_dictionary, load_dictionary, serialize_dictionary, stack
from src.training_evaluation import Statistics, MultipleRunStatistics, ModelParameters

class Trainer():    
    def __init__(self, test_data):
        self.observations, self.protected_attributes, self.ground_truths = test_data
        
    def _training_iteration(self, training_parameters):
        np.random.seed()

        decisions_over_time = None
        parameters_over_time = []
        lagrangians_over_time = []

        for policy in consequential_learning(**training_parameters):
            # Store decisions made in last time step ...
            decisions = policy(self.observations, self.protected_attributes).reshape(-1, 1)
            decisions_over_time = stack(decisions_over_time, decisions, axis=1)
            
            # ... and the parameters of the model
            parameters_over_time.append(policy.get_model_parameters())
            lagrangians_over_time.append(policy.get_lagrangian_multiplier())

        return decisions_over_time, parameters_over_time, np.array(lagrangians_over_time)

    def train_over_iterations(self, training_parameters, iterations, asynchronous):
        decisions_tensor = None
        model_parameters = ModelParameters()
        # multithreaded runs of training
        if asynchronous:
            apply_results = []
            pool = mp.Pool(mp.cpu_count())
            for iteration in range(0, iterations):
                apply_results.append((iteration, pool.apply_async(self._training_iteration, args=(training_parameters,))))
            pool.close()
            pool.join()

            for result in apply_results:
                decisions_over_time, parameters_over_time, lagrangians_over_time = result[1].get()
                decisions_tensor = stack(decisions_tensor, decisions_over_time, axis=2)
                model_parameters.add(result[0], parameters_over_time, lagrangians_over_time)
        else:
            for iteration in range(0, iterations):
                decisions_over_time, parameters_over_time, lagrangians_over_time = self._training_iteration(training_parameters)
                decisions_tensor = stack(decisions_tensor, decisions_over_time, axis=2)
                model_parameters.add(iteration, parameters_over_time, lagrangians_over_time)

        return Statistics.calculate_statistics(
                predictions=decisions_tensor, 
                observations=self.observations,
                protected_attributes=self.protected_attributes, 
                ground_truths=self.ground_truths, 
                utility_function=training_parameters["model"]["utility_function"]), model_parameters

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

def train(training_parameters, fairness_rates=None, iterations=30, asynchronous=True):
    """ Executes multiple runs of consequential learning with the same training parameters
    but different seeds for the specified fairness rates. 
        
    Args:
        training_parameters: The parameters used to configure the consequential learning algorithm.
        fairness_rates: An iterable containing all fairness rates for which consequential learning
        should be run and statistics will be collected. If fairness_rate=None then no preset value
        for the fairness function is assumed and a full lagrange multiplier is executed.
        iterations: The number of times consequential learning will be run for one of the specified
        fairness rates. The resulting statistics will be applied over the number of runs
        asynchronous: A flag indicating if the iterations should be executed asynchronously.

    Returns:
        training_statistic: A dictionary that contains statistical data about 
        the executed runs.
    """
    current_training_parameters = deepcopy(training_parameters)

    if "save_path" in training_parameters:
        base_save_path = "{}/runs".format(training_parameters["save_path"])
        Path(base_save_path).mkdir(parents=True, exist_ok=True)
        runs = os.listdir(base_save_path)

        if len(runs) == 0:
            current_run = 0
        else:
            runs.sort(key=int)
            current_run = int(runs[-1]) + 1

        base_save_path = "{}/{}".format(base_save_path, current_run)
        Path(base_save_path).mkdir(parents=True, exist_ok=True)
        parameter_save_path = "{}/parameters.json".format(base_save_path)

        current_training_parameters["lambdas"] = fairness_rates
        serialized_dictionary = serialize_dictionary(current_training_parameters)
        save_dictionary(serialized_dictionary, parameter_save_path)
    else:
        base_save_path = None

    current_training_parameters["data"] = _generate_data_set(training_parameters)

    if base_save_path is not None:
        data_save_path = "{}/data.json".format(base_save_path)
        data_dict = {
            "x": current_training_parameters["data"]["test_dataset"][0].tolist(),
            "s": current_training_parameters["data"]["test_dataset"][1].tolist(),
            "y": current_training_parameters["data"]["test_dataset"][2].tolist()
        }
        save_dictionary(data_dict, data_save_path)

    trainer = Trainer(test_data=current_training_parameters["data"]["test_dataset"])

    if fairness_rates is None:
        fairness_rates = [None]

    overall_statistics = MultipleRunStatistics()
    for fairness_rate in fairness_rates:
        print("Processing Lambda {} ".format(fairness_rate))

        current_training_parameters["optimization"]["fairness_rate"] = fairness_rate
        statistics, model_parameters = trainer.train_over_iterations(current_training_parameters, iterations, asynchronous)

        if len(fairness_rates) == 1:
            overall_statistics = statistics
        else:
            overall_statistics.log_run(statistics=statistics, fairness_rate=fairness_rate)

        if base_save_path is not None:
            # save the last theta for all iterations of a specified lambda:
            lambda_path = "{}/lambda{}/".format(base_save_path, fairness_rate)
            Path(lambda_path).mkdir(parents=True, exist_ok=True)
            model_save_path = "{}models.json".format(lambda_path)

            serialized_model_parameters = serialize_dictionary(model_parameters.to_dict())

            save_dictionary(serialized_model_parameters, model_save_path)

    return overall_statistics, base_save_path
