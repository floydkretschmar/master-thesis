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

from src.consequential_learning import ConsequentialLearning, FixedLambdasConsequentialLearning, DualGradientConsequentialLearning
from src.util import save_dictionary, load_dictionary, serialize_dictionary, stack, check_for_missing_kwargs
from src.training_evaluation import Statistics, MultiStatistics, ModelParameters

class _Trainer():
    def _training_iteration(self, training_parameters, training_method):
        np.random.seed()
        results = []
        for result in training_method(training_parameters):
            results.append(deepcopy(result))
        return results

    @staticmethod
    def _process_results(results_per_iterations):
        decision_tensors = []
        model_parameters_over_lambdas = {}
        for num_iteration, iteration_result in enumerate(results_per_iterations):
            for num_lambda, lambda_result in enumerate(iteration_result):
                decisions_over_time, model_parameters = lambda_result

                if num_lambda >= len(decision_tensors):
                    decision_tensors.append(decisions_over_time)
                else:
                    decision_tensors[num_lambda] = stack(decision_tensors[num_lambda], decisions_over_time, axis=2)
                
                if num_lambda in model_parameters_over_lambdas:
                    model_parameters_over_lambdas[num_lambda][num_iteration] = model_parameters
                else:
                    model_parameters_over_lambdas[num_lambda] = {
                        num_iteration: model_parameters
                    }
                    
        return decision_tensors, model_parameters_over_lambdas

    @staticmethod
    def _partial_results(training_parameters, results_per_iterations, total_statistics, total_parameters):
        decision_tensors, model_parameters = _Trainer._process_results(results_per_iterations)

        for num_lambda, decisions_tensor in enumerate(decision_tensors):
            statistics = Statistics.calculate_statistics(
                    predictions=decisions_tensor, 
                    observations=training_parameters["data"]["test"][0],
                    protected_attributes=training_parameters["data"]["test"][1], 
                    ground_truths=training_parameters["data"]["test"][2], 
                    utility_function=training_parameters["model"]["utility_function"])

            if num_lambda < len(total_statistics):
                total_statistics[num_lambda] = total_statistics[num_lambda].merge(statistics)  
            else:
                total_statistics.append(statistics)

            if num_lambda in total_parameters:
                offset = max(total_parameters[num_lambda], key=int) + 1

                for (iteration, iteration_value) in model_parameters[num_lambda].items():
                    total_parameters[num_lambda][offset + iteration] = iteration_value
            else:
                total_parameters[num_lambda] = model_parameters[num_lambda]   
        
        del model_parameters, decision_tensors
        return total_statistics, total_parameters

    def train_over_iterations(self, training_parameters, training_method, iterations, iteration_split, asynchronous):        
        results_per_iterations = []

        total_statistics = []
        total_parameters = {}
        # multithreaded runs of training
        if asynchronous:
            for start in range(0, iterations, iteration_split):
                apply_results = []
                results_per_iterations = []
                pool = mp.Pool(mp.cpu_count())
                for iteration in range(0, iteration_split):
                    apply_results.append(pool.apply_async(self._training_iteration, args=(training_parameters, training_method)))
                pool.close()
                pool.join()
                
                for result in apply_results:
                    results_per_iterations.append(result.get())
                    
                total_statistics, total_parameters = _Trainer._partial_results(training_parameters, results_per_iterations, total_statistics, total_parameters)
                del results_per_iterations, apply_results

                print("Finished iteration {} to {}".format(start, start+iteration_split))
        else:
            for iteration in range(0, iterations):
                results_per_iterations.append(self._training_iteration(training_parameters, training_method))

                if (iteration % iteration_split == 0 and iteration != 0) or (iteration+1 == iterations and len(results_per_iterations) > 0):
                    total_statistics, total_parameters = _Trainer._partial_results(training_parameters, results_per_iterations, total_statistics, total_parameters)
                    del results_per_iterations
            
        if len(total_statistics) == 1:
            return total_statistics[0], total_parameters

        return total_statistics, total_parameters


def _check_for_missing_training_parameters(training_parameters):
    check_for_missing_kwargs("training()", ["experiment_name", "model", "parameter_optimization", "data"], training_parameters)
    check_for_missing_kwargs(
        "training()", 
        ["benefit_function", "utility_function", "fairness_function", "feature_map", "learn_on_entire_history", "use_sensitve_attributes", "bias"], 
        training_parameters["model"])
    check_for_missing_kwargs("training()", ["distribution", "fraction_protected", "num_test_samples", "num_decisions"], training_parameters["data"])
    check_for_missing_kwargs("training()", ["time_steps", "parameters"], training_parameters["optimization"])
    check_for_missing_kwargs("training()", ["lambda", "theta"], training_parameters["optimization"]["parameters"])


def _generate_data_set(training_parameters):
    """ Generates one training and test dataset to be used across all lambdas.
        
    Args:
        training_parameters: The parameters used to configure the consequential learning algorithm.

    Returns:
        data: A dictionary containing both the test and training dataset.
    """
    num_decisions = training_parameters["data"]["num_decisions"]
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
        data["training"]["lambda"] = distribution.sample_dataset(
            n=num_decisions, 
            fraction_protected=fraction_protected)
        # lambda_train_x, lambda_train_s, lambda_train_y = distribution.sample_dataset(
        # n=num_decisions * training_parameters["lagrangian_optimization"]["iterations"], 
        # fraction_protected=fraction_protected)

        #lambda_train_datasets = []
        #for i in range(0, lambda_train_x.shape[0], num_decisions):
        #    lambda_train_datasets.append((lambda_train_x[i:i+num_decisions], lambda_train_s[i:i+num_decisions], lambda_train_y[i:i+num_decisions]))

        #data["training"]["lambda"] = lambda_train_datasets

    return data
           
     
def _save_results(model_parameters, statistics, base_save_path, sub_directory=None):
    # save the model parameters to be able to restore the model
    if sub_directory is not None:
        lambda_path = "{}/{}/".format(base_save_path, sub_directory)
        Path(lambda_path).mkdir(parents=True, exist_ok=True)
    else:
        lambda_path = "{}/".format(base_save_path)

    model_save_path = "{}models.json".format(lambda_path)

    serialized_model_parameters = serialize_dictionary(model_parameters)
    save_dictionary(serialized_model_parameters, model_save_path)

    # save the results for each lambda
    statistics_save_path = "{}statistics.json".format(lambda_path)
    serialized_statistics = serialize_dictionary(statistics.to_dict())
    save_dictionary(serialized_statistics, statistics_save_path)

def train(training_parameters, iterations=30, iteration_split=10, asynchronous=True):
    """ Executes multiple runs of consequential learning with the same training parameters
    but different seeds for the specified fairness rates. 
        
    Args:
        training_parameters: The parameters used to configure the consequential learning algorithm.
        iterations: The number of times consequential learning will be run for one of the specified
        fairness rates. The resulting statistics will be applied over the number of runs
        asynchronous: A flag indicating if the iterations should be executed asynchronously.

    Returns:
        training_statistic: A dictionary that contains statistical data about 
        the executed runs.
        model_parameters: A single ModelParameters object if training both lambda and theta,
        or None, when training with fixed lambdas. The ModelParameter object contains theta 
        and lambda values for each timestep over all iterations.
    """
    assert iterations > 0
    #_check_for_missing_training_parameters(training_parameters)
    current_training_parameters = deepcopy(training_parameters)

    if "save" in training_parameters and training_parameters["save"]:
        base_save_path = "{}/res/{}".format(root_path, training_parameters["experiment_name"])
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

    trainer = _Trainer()

    if isinstance(current_training_parameters["model"]["initial_lambda"], numbers.Number) and "lagrangian_optimization" not in training_parameters:
        print("---------- Single training run for fixed lambda ----------")
        training_algorithm = ConsequentialLearning(current_training_parameters["model"]["learn_on_entire_history"])
        overall_statistics, model_parameters = trainer.train_over_iterations(current_training_parameters, training_algorithm.train, iterations, iteration_split, asynchronous)
    else:
        # if list of parameters is given we fix lambda over all time steps and train vor all lambdas
        if not isinstance(current_training_parameters["model"]["initial_lambda"], numbers.Number):
            print("---------- Training with fixed lambdas ----------")
            fairness_rates = deepcopy(current_training_parameters["model"]["initial_lambda"])
            training_algorithm = FixedLambdasConsequentialLearning(current_training_parameters["model"]["learn_on_entire_history"])
            sub_directories = ["lambda_{}".format(fairness_rate) for fairness_rate in fairness_rates]
            overall_statistics = MultiStatistics("log", fairness_rates, "Lambda")
        # if a dict is given for lambda we try to learn the perfect lambda
        elif "lagrangian_optimization" in training_parameters:
            print("---------- Training both theta and lambda ----------")
            training_algorithm = DualGradientConsequentialLearning(current_training_parameters["model"]["learn_on_entire_history"])
            sub_directories = ["epoch_{}".format(epoch) for epoch in range(0, current_training_parameters["lagrangian_optimization"]["iterations"])]
            overall_statistics = MultiStatistics("linear", range(0, current_training_parameters["lagrangian_optimization"]["iterations"]), "Lambda Training Iteration")
            
        statistics, model_parameters = trainer.train_over_iterations(current_training_parameters, training_algorithm.train, iterations, iteration_split, asynchronous)

        for num_lambda, statistic in enumerate(statistics):
            overall_statistics.log_run(statistic)

            if base_save_path is not None:  
                _save_results(model_parameters[num_lambda], statistic, base_save_path, sub_directory=sub_directories[num_lambda])
            
    # save the overall results
    if base_save_path is not None:
        overall_stat_save_path = "{}/overall_statistics.json".format(base_save_path)
        serialized_statistics = serialize_dictionary(overall_statistics.to_dict())
        save_dictionary(serialized_statistics, overall_stat_save_path)

    return overall_statistics, ModelParameters(model_parameters), base_save_path
    
