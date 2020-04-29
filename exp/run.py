import argparse
import multiprocessing as mp
import os
import sys
from copy import deepcopy
from queue import Queue

import numpy as np
from pathos.pools import _ThreadPool as Pool

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.feature_map import IdentityFeatureMap
from src.functions import cost_utility
from src.plotting import plot_mean, plot_median
from src.training import train
from src.distribution import FICODistribution, COMPASDistribution, AdultCreditDistribution
from src.util import mean_difference


# region Fairness Definitions

def calc_benefit(decisions, ips_weights):
    if ips_weights is not None:
        decisions *= ips_weights

    return decisions


def calc_covariance(s, decisions, ips_weights):
    new_s = 1 - (2 * s)

    if ips_weights is not None:
        mu_s = np.mean(new_s * ips_weights, axis=0)
        d = decisions * ips_weights
    else:
        mu_s = np.mean(new_s, axis=0)
        d = decisions

    covariance = (new_s - mu_s) * d
    return covariance


def fairness_function_gradient(type, **fairness_kwargs):
    policy = fairness_kwargs["policy"]
    x = fairness_kwargs["x"]
    s = fairness_kwargs["s"]
    y = fairness_kwargs["y"]
    decisions = fairness_kwargs["decisions"]
    ips_weights = fairness_kwargs["ips_weights"]

    if type == "BD_DP" or type == "BD_EOP":
        result = calc_benefit(decisions, ips_weights)
    elif type == "COV_DP":
        result = calc_covariance(s, decisions, ips_weights)

    log_gradient = policy.log_policy_gradient(x, s)
    grad = log_gradient * result

    if type == "BD_DP":
        return mean_difference(grad, s)
    elif type == "COV_DP":
        return np.mean(grad, axis=0)
    elif type == "BP_EOP":
        y1_indices = np.where(y == 1)
        return mean_difference(grad[y1_indices], s[y1_indices])


def fairness_function(type, **fairness_kwargs):
    s = fairness_kwargs["s"]
    decisions = fairness_kwargs["decisions"]
    ips_weights = fairness_kwargs["ips_weights"]
    y = fairness_kwargs["y"]

    if type == "BD_DP":
        benefit = calc_benefit(decisions, ips_weights)
        return mean_difference(benefit, s)
    elif type == "COV_DP":
        covariance = calc_covariance(s, decisions, ips_weights)
        return np.mean(covariance, axis=0)
    elif type == "BP_EOP":
        benefit = calc_benefit(decisions, ips_weights)
        y1_indices = np.where(y == 1)
        return mean_difference(benefit[y1_indices], s[y1_indices])


# endregion

parser = argparse.ArgumentParser()

parser.add_argument('-d', '--data', type=str, required=True, help="select the distribution (FICO, COMPAS, ADULT)")
parser.add_argument('-c', '--cost', type=float, required=True, help="define the utility cost c")
parser.add_argument('-lr', '--learning_rate', type=float, required=True, help="define the learning rate of theta")
parser.add_argument('-p', '--path', type=str, required=False, help="save path for the result")
parser.add_argument('-ts', '--time_steps', type=int, nargs='+', required=True, help='list of time steps to be used')
parser.add_argument('-e', '--epochs', type=int, nargs='+', required=True, help='list of epochs to be used')
parser.add_argument('-bs', '--batch_sizes', type=int, nargs='+', required=True, help='list of batch sizes to be used')
parser.add_argument('-nb', '--num_batches', type=int, nargs='+', required=True,
                    help='list of number of batches to be used')
parser.add_argument('-i', '--iterations', type=int, required=True, help='the number of internal iterations')
parser.add_argument('-a', '--asynchronous', action='store_true')
parser.add_argument('--plot', required=False, action='store_true')
parser.add_argument('-pid', '--process_id', type=str, required=False, help="process id for identification")

parser.add_argument('-f', '--fairness_type', type=str, required=False,
                    help="select the type of fairness (BD_DP, COV_DP, BP_EOP). "
                         "if none is selected no fairness criterion is applied")
parser.add_argument('-fl', '--fairness_lower_bound', type=float, required=False, help='the lowest value for lambda')
parser.add_argument('-fu', '--fairness_upper_bound', type=float, required=False, help='the highest value for lambda')

args = parser.parse_args()

if args.fairness_type:
    if args.fairness_lower_bound is None:
        parser.error('when using --fairness_type, --fairness_lower_bount has to be specified')
    else:
        fair_fct = lambda **fairness_params: fairness_function(type=args.fairness_type, **fairness_params)
        fair_fct_grad = lambda **fairness_params: fairness_function_gradient(type=args.fairness_type, **fairness_params)

    if args.fairness_upper_bound is not None:
        initial_lambda = np.geomspace(args.fairness_lower_bound, args.fairness_upper_bound, endpoint=True, num=20)
    else:
        initial_lambda = args.fairness_lower_bound
else:
    fair_fct = lambda **fairness_params: [0.0]
    fair_fct_grad = lambda **fairness_params: [0.0]
    initial_lambda = 0.0

if args.data == 'FICO':
    distibution = FICODistribution(bias=True, fraction_protected=0.5)
elif args.data == 'COMPAS':
    distibution = COMPASDistribution(bias=True, test_percentage=0.2)
elif args.data == 'ADULT':
    distibution = AdultCreditDistribution(bias=True, test_percentage=0.2)

training_parameters = {
    'distribution': distibution,
    'model': {
        'fairness_function': fair_fct,
        'fairness_gradient_function': fair_fct_grad,
        'utility_function': lambda **util_params: cost_utility(cost_factor=args.cost, **util_params),
        'feature_map': IdentityFeatureMap(distibution.feature_dimension),
        'learn_on_entire_history': False,
        'use_sensitve_attributes': False,
        'bias': True,
        'initial_theta': np.zeros(distibution.feature_dimension),
        'initial_lambda': initial_lambda
    },
    'parameter_optimization': {
        'learning_rate': args.learning_rate,
        'decay_rate': 1,
        'decay_step': 10000,
        'fix_seeds': True
    },
    'test': {
        'num_samples': 10000
    }
}

callback_queue = Queue()
pool = Pool(mp.cpu_count())
thread_count = 0

for time_steps in args.time_steps:
    training_parameters['parameter_optimization']['time_steps'] = time_steps
    for epochs in args.epochs:
        training_parameters['parameter_optimization']['epochs'] = epochs
        for batch_size in args.batch_sizes:
            training_parameters['parameter_optimization']['batch_size'] = batch_size
            for num_batches in args.num_batches:
                training_parameters['parameter_optimization']['num_batches'] = num_batches

                if args.path:
                    training_parameters['save_path'] = args.path
                    training_parameters["save_path"] = "{}/c{}/lr{}/ts{}-ep{}-bs{}-nb{}".format(args.path,
                                                                                                args.cost,
                                                                                                args.learning_rate,
                                                                                                time_steps,
                                                                                                epochs,
                                                                                                batch_size,
                                                                                                num_batches)
                    training_parameters["save_path_subfolder"] = args.process_id

                pool.apply_async(train,
                                 args=(deepcopy(training_parameters), args.iterations, args.asynchronous),
                                 callback=lambda result: callback_queue.put(result),
                                 error_callback=lambda e: print(e))
                thread_count += 1

best_final_median_utility = float("-inf")
best_utility_path = ""
while thread_count > 0:
    result = callback_queue.get()
    statistics, model_parameters, run_path = result

    final_median_utility = statistics.performance(statistics.UTILITY, statistics.MEDIAN)[-1]

    if final_median_utility > best_final_median_utility:
        best_final_median_utility = final_median_utility
        best_utility_path = run_path

    if args.plot:
        plot_mean(statistics, "{}/results_mean_time.png".format(run_path))
        plot_median(statistics, "{}/results_median_time.png".format(run_path))

    thread_count -= 1

print(best_utility_path)

pool.close()
pool.join()
