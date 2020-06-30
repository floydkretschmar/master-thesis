import os
import sys

root_path = os.path.abspath(os.path.join("."))
if root_path not in sys.path:
    sys.path.append(root_path)

import time
from pathlib import Path
from copy import deepcopy

from src.consequential_learning import ConsequentialLearning
from src.util import save_dictionary, serialize_dictionary, check_for_missing_kwargs, fix_seed
from src.training_evaluation import MultiStatistics


def _check_for_missing_training_parameters(training_parameters):
    """ Checks the dictionary of training parameters specified by the user for missing entries.
        
    Args:
        training_parameters: The parameters used to configure the consequential learning algorithm.
    """
    check_for_missing_kwargs("training()",
                             ["model", "distribution", "optimization_target", "parameter_optimization", "data"],
                             training_parameters)

    check_for_missing_kwargs("training()",
                             ["batch_size", "epochs", "learning_rate", "learn_on_entire_history", "time_steps", "training_algorithm"],
                             training_parameters["parameter_optimization"])

    check_for_missing_kwargs("training()", ["num_test_samples", "num_train_samples"],
                             training_parameters["data"])

    if "lagrangian_optimization" in training_parameters:
        check_for_missing_kwargs("training()",
                                 ["batch_size", "epochs", "learning_rate", "training_algorithm"],
                                 training_parameters["lagrangian_optimization"])


def _prepare_training(training_parameters):
    """ Preprocesses the training parameters and sets up the training procedure
        
    Args:
        training_parameters: The parameters used to configure the consequential learning algorithm.

    Returns:
        current_training_parameters: The pre processed training parameters.
    """
    _check_for_missing_training_parameters(training_parameters)
    current_training_parameters = deepcopy(training_parameters)

    ##################### SAVE PARAMETER SETTINGS #####################
    if "save_path" in training_parameters:
        # base_save_path = "{}/{}".format(training_parameters["save_path"], training_parameters["experiment_name"])
        base_save_path = training_parameters["save_path"]
        Path(base_save_path).mkdir(parents=True, exist_ok=True)

        timestamp = time.gmtime()

        if "save_path_subfolder" in training_parameters:
            ts_folder = training_parameters["save_path_subfolder"]
        else:
            ts_folder = time.strftime("%Y-%m-%d-%H-%M-%S", timestamp)

        base_save_path = "{}/{}".format(base_save_path, ts_folder)
        Path(base_save_path).mkdir(parents=True, exist_ok=True)
        parameter_save_path = "{}/parameters.json".format(base_save_path)

        serialized_dictionary = serialize_dictionary(current_training_parameters)
        save_dictionary(serialized_dictionary, parameter_save_path)
    else:
        base_save_path = None

    # if decay_rate and decay_step not specified: set them in a way such that there is no decay of the lr
    if "decay_rate" not in training_parameters["parameter_optimization"]:
        current_training_parameters["parameter_optimization"]["decay_rate"] = 1

    if "decay_step" not in training_parameters["parameter_optimization"]:
        current_training_parameters["parameter_optimization"]["decay_step"] \
            = training_parameters["parameter_optimization"]["time_steps"] + 1

    # if epochs, change_iterations or change percentage is not specified, set to default values
    current_training_parameters["parameter_optimization"]["epochs"] = training_parameters["parameter_optimization"][
        "epochs"] if "epochs" in training_parameters["parameter_optimization"] else None

    current_training_parameters["parameter_optimization"]["change_iterations"] = \
        training_parameters["parameter_optimization"]["change_iterations"] \
            if "change_iterations" in training_parameters["parameter_optimization"] else 5

    current_training_parameters["parameter_optimization"]["change_percentage"] = \
        training_parameters["parameter_optimization"]["change_percentage"] \
            if "change_percentage" in training_parameters["parameter_optimization"] else 0.05

    # if weight clipping is not specified set to false
    if "clip_weights" not in current_training_parameters["parameter_optimization"]:
        current_training_parameters["parameter_optimization"]["clip_weights"] = False

    ##################### PERPARE LAGRANGIAN OPTIMIZATION #####################
    if "lagrangian_optimization" in current_training_parameters:
        # if decay_rate and decay_step not specified: set them in a way such that there is no decay of the lr
        if "decay_rate" not in training_parameters["lagrangian_optimization"]:
            current_training_parameters["lagrangian_optimization"]["decay_rate"] = 1

        if "decay_step" not in training_parameters["lagrangian_optimization"]:
            current_training_parameters["lagrangian_optimization"]["decay_step"] \
                = training_parameters["parameter_optimization"]["time_steps"] + 1

    current_training_parameters["save_path"] = base_save_path
    return current_training_parameters, base_save_path


def merge_run_results(results_per_run):
    overall_statistics, overall_model_parameters = results_per_run[0]

    for run, run_result in enumerate(results_per_run[1:]):
        statistics, model_parameters = run_result
        overall_statistics.merge(statistics)
        overall_model_parameters.merge(model_parameters)

    return overall_statistics, overall_model_parameters


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
        lambda_path = "{}/runs/{}/".format(base_save_path, sub_directory)
        Path(lambda_path).mkdir(parents=True, exist_ok=True)
    else:
        lambda_path = "{}/".format(base_save_path)

    if model_parameters is not None:
        model_save_path = "{}models.json".format(lambda_path)

        serialized_model_parameters = serialize_dictionary(model_parameters.to_dict())
        save_dictionary(serialized_model_parameters, model_save_path)

    # save the results for each lambda
    statistics_save_path = "{}statistics.json".format(lambda_path)
    serialized_statistics = statistics.to_dict()
    save_dictionary(serialized_statistics, statistics_save_path)


def train(training_parameters, fairness_rates=[0.0]):
    """ Executes multiple runs of consequential learning with the same training parameters
    but different seeds for the specified fairness rates. 
        
    Args:
        training_parameters: The parameters used to configure the consequential learning algorithm.
        iterations: The number of times consequential learning will be run for one of the specified or a list of seed
                    for each of which one run will be started.
        asynchronous: A flag indicating if the iterations should be executed asynchronously.

    Returns:
        training_statistic: A dictionary that contains statistical data about 
        the executed runs.
        model_parameters: A single ModelParameters object if training both lambda and theta,
        or None, when training with fixed lambdas. The ModelParameter object contains theta 
        and lambda values for each timestep over all iterations.
    """
    #fix_seed(seed)
    current_training_parameters, base_save_path = _prepare_training(training_parameters)
    multiple_lambdas = len(fairness_rates) > 1

    if multiple_lambdas:
        overall_statistics = MultiStatistics()

    for fairness_rate in fairness_rates:
        info_string = "// LR = {} // TS = {} // E = {} // BS = {} // FR = {}".format(
            current_training_parameters["parameter_optimization"]["learning_rate"],
            current_training_parameters["parameter_optimization"]["time_steps"],
            current_training_parameters["parameter_optimization"]["epochs"],
            current_training_parameters["parameter_optimization"]["batch_size"],
            fairness_rate)
        print("## STARTED {} ##".format(info_string))
        current_training_parameters["optimization_target"].fairness_rate = fairness_rate

        training_algorithm = ConsequentialLearning(
            training_parameters["parameter_optimization"]["learn_on_entire_history"])
        statistics, model_parameters = training_algorithm.train(current_training_parameters)

        if multiple_lambdas:
            overall_statistics.log_run(statistics)

        if base_save_path is not None:
            _save_results(
                base_save_path=base_save_path,
                statistics=statistics,
                model_parameters=model_parameters,
                sub_directory="lambda_{}".format(fairness_rate) if multiple_lambdas else None)

        print("## ENDED {} ##".format(info_string))

    # save the overall results    
    if base_save_path is not None and multiple_lambdas:
        _save_results(base_save_path=base_save_path,
                      statistics=overall_statistics)

    return overall_statistics if multiple_lambdas else statistics, model_parameters, base_save_path
