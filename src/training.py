import os
import sys
root_path = os.path.abspath(os.path.join('.'))
if root_path not in sys.path:
    sys.path.append(root_path)

import multiprocessing as mp
from pathos.multiprocessing import ProcessingPool as Pool
import time
from pathlib import Path
from copy import deepcopy
import numbers

from src.consequential_learning import ConsequentialLearning, DualGradientConsequentialLearning
from src.util import save_dictionary, serialize_dictionary, check_for_missing_kwargs, get_list_of_seeds
from src.training_evaluation import Statistics, MultiStatistics, ModelParameters
from src.optimization import FairnessFunction, UtilityFunction


class _Trainer():
    def _training_iteration(self, training_parameters, training_method):
        # np.random.seed()
        results_over_lambdas = []
        x_test, s_test, y_test = training_parameters["test"]

        for results, model_parameters in training_method(training_parameters):
            decisions_over_time, fairness_over_time, utility_over_time = results
            statistics = Statistics(
                predictions=decisions_over_time,
                observations=x_test,
                fairness=fairness_over_time,
                utility=utility_over_time,
                protected_attributes=s_test,
                ground_truths=y_test)
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
        # multithreaded runs of training
        if asynchronous:
            apply_results = []
            results_per_iterations = []
            pool = Pool(mp.cpu_count())
            for _ in range(0, iterations):
                # apply_results.append(pool.apply_async(self._training_iteration, args=(training_parameters, training_method)))
                apply_results.append(pool.apipe(self._training_iteration, training_parameters, training_method))
            # pool.close()
            # pool.join()

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
    check_for_missing_kwargs("training()",
                             ["experiment_name", "model", "parameter_optimization", "test", "distribution"],
                             training_parameters)
    check_for_missing_kwargs(
        "training()",
        ["utility_function", "fairness_function", "fairness_gradient_function", "feature_map",
         "learn_on_entire_history", "use_sensitve_attributes", "bias", "initial_theta", "initial_lambda"],
        training_parameters["model"])
    check_for_missing_kwargs("training()", ["num_samples"], training_parameters["test"])
    check_for_missing_kwargs("training()",
                             ["time_steps", "epochs", "batch_size", "learning_rate", "decay_rate", "decay_step",
                              "num_batches"], training_parameters["parameter_optimization"])

    if "lagrangian_optimization" in training_parameters:
        check_for_missing_kwargs("training()",
                                 ["epochs", "batch_size", "learning_rate", "decay_rate", "decay_step", "num_batches"],
                                 training_parameters["lagrangian_optimization"])


def _prepare_training(training_parameters):
    """ Preprocesses the training parameters and sets up the training procedure
        
    Args:
        training_parameters: The parameters used to configure the consequential learning algorithm.

    Returns:
        current_training_parameters: The pre processed training parameters.
    """
    _check_for_missing_training_parameters(training_parameters)
    current_training_parameters = training_parameters

    # save parameter settings
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

    # if fixed seeding for parameter optimization: get one seed per time step for data generation
    if "fix_seeds" in current_training_parameters["parameter_optimization"] \
            and current_training_parameters["parameter_optimization"]["fix_seeds"]:
        current_training_parameters["parameter_optimization"]["seeds"] = get_list_of_seeds(
            training_parameters['parameter_optimization']['time_steps'])

    # if fixed seeding for parameter lagrangian_optimization: get one seed per iteration for data generation
    if "lagrangian_optimization" in current_training_parameters \
            and "fix_seeds" in current_training_parameters["lagrangian_optimization"] \
            and current_training_parameters["lagrangian_optimization"]["fix_seeds"]:
        current_training_parameters["lagrangian_optimization"]["seeds"] = get_list_of_seeds(
            training_parameters['lagrangian_optimization']['iterations'])

    # generate one set of test data across all threads in advance
    current_training_parameters["test"] = training_parameters["distribution"].sample_test_dataset(
        n_test=training_parameters["test"]["num_samples"])

    # convert utility and fairness functions into appropriate internal functions
    current_training_parameters["model"]["utility_function"] = UtilityFunction(
        training_parameters["model"]["utility_function"])

    if "fairness_gradient_function" in training_parameters["model"]:
        current_training_parameters["model"]["fairness_function"] = FairnessFunction(
            training_parameters["model"]["fairness_function"],
            training_parameters["model"]["fairness_gradient_function"]
        )

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
        lambda_path = "{}/runs/{}/".format(base_save_path, sub_directory)
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

    if isinstance(current_training_parameters["model"]["initial_lambda"],
                  numbers.Number) and "lagrangian_optimization" not in training_parameters:
        info_string = "// LR = {} // TS = {} // E = {} // BS = {} // NB = {}".format(
            current_training_parameters["parameter_optimization"]["learning_rate"],
            current_training_parameters["parameter_optimization"]["time_steps"],
            current_training_parameters["parameter_optimization"]["epochs"],
            current_training_parameters["parameter_optimization"]["batch_size"],
            current_training_parameters["parameter_optimization"]["num_batches"])
        print("## STARTED Single training run {} ##".format(info_string))
        training_algorithm = ConsequentialLearning(current_training_parameters["model"]["learn_on_entire_history"])
        overall_statistics, model_parameters = trainer.train_over_iterations(current_training_parameters,
                                                                             training_algorithm.train, iterations,
                                                                             asynchronous)
        print("## ENDED Single training run {} ##".format(info_string))
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
            statistics=overall_statistics,
            model_parameters=model_parameters)

    return overall_statistics, ModelParameters(model_parameters), base_save_path
    
