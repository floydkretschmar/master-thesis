import argparse
import os
import sys
from copy import deepcopy

import numpy as np

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
    elif type == "BD_EOP":
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
    elif type == "BD_EOP":
        benefit = calc_benefit(decisions, ips_weights)
        y1_indices = np.where(y == 1)
        return mean_difference(benefit[y1_indices], s[y1_indices])


# endregion

parser = argparse.ArgumentParser()

parser.add_argument('-d', '--data', type=str, required=True, help="select the distribution (FICO, COMPAS, ADULT)")
parser.add_argument('-c', '--cost', type=float, required=True, help="define the utility cost c")
parser.add_argument('-lr', '--learning_rate', type=float, required=True, help="define the learning rate of theta")
parser.add_argument('-p', '--path', type=str, required=False, help="save path for the result")
parser.add_argument('-ts', '--time_steps', type=int, required=True, help='number of time steps to be used')
parser.add_argument('-e', '--epochs', type=int, required=True, help='number of epochs to be used')
parser.add_argument('-bs', '--batch_size', type=int, required=True, help='batch size to be used')
parser.add_argument('-nb', '--num_batches', type=int, required=True, help='number of batches to be used')
parser.add_argument('-i', '--iterations', type=int, required=True, help='the number of internal iterations')
parser.add_argument('-a', '--asynchronous', action='store_true')
parser.add_argument('--plot', required=False, action='store_true')
parser.add_argument('-pid', '--process_id', type=str, required=False, help="process id for identification")

parser.add_argument('-f', '--fairness_type', type=str, required=False,
                    help="select the type of fairness (BD_DP, COV_DP, BD_EOP). "
                         "if none is selected no fairness criterion is applied")
parser.add_argument('-fv', '--fairness_value', type=float, required=False, help='the value of lambda')

args = parser.parse_args()

if args.fairness_type:
    if args.fairness_value is None:
        parser.error('when using --fairness_type, --fairness_value has to be specified')
    else:
        fair_fct = lambda **fairness_params: fairness_function(type=args.fairness_type, **fairness_params)
        fair_fct_grad = lambda **fairness_params: fairness_function_gradient(type=args.fairness_type, **fairness_params)
        initial_lambda = args.fairness_value
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

training_parameters['parameter_optimization']['time_steps'] = args.time_steps
training_parameters['parameter_optimization']['epochs'] = args.epochs
training_parameters['parameter_optimization']['batch_size'] = args.batch_size
training_parameters['parameter_optimization']['num_batches'] = args.num_batches

if args.path:
    if args.fairness_type is not None:
        training_parameters["save_path"] = "{}/c{}/lr{}/ts{}-ep{}-bs{}-nb{}".format(args.path,
                                                                                    args.cost,
                                                                                    args.learning_rate,
                                                                                    args.time_steps,
                                                                                    args.epochs,
                                                                                    args.batch_size,
                                                                                    args.num_batches)

        if args.process_id is not None:
            training_parameters["save_path_subfolder"] = "{}/{}".format(args.fairness_value, args.process_id)
        else:
            training_parameters["save_path_subfolder"] = args.fairness_value
    else:
        training_parameters["save_path"] = "{}/no_fairness/c{}/lr{}/ts{}-ep{}-bs{}-nb{}".format(args.path,
                                                                                                args.cost,
                                                                                                args.learning_rate,
                                                                                                args.time_steps,
                                                                                                args.epochs,
                                                                                                args.batch_size,
                                                                                                args.num_batches)
        if args.process_id is not None:
            training_parameters["save_path_subfolder"] = args.process_id

statistics, model_parameters, run_path = train(deepcopy(training_parameters), args.iterations, args.asynchronous)

if args.plot:
    plot_mean(statistics, "{}/results_mean_time.png".format(run_path))
    plot_median(statistics, "{}/results_median_time.png".format(run_path))
