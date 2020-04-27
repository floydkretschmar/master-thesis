import os
import sys

import numpy as np

module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)

from src.util import mean_difference
from src.feature_map import IdentityFeatureMap
from src.functions import cost_utility
from src.distribution import FICODistribution, COMPASDistribution, AdultCreditDistribution


# ============================================= DEFINE FARINESS FCT ================================================== #

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


training_parameters = {
    'save_path': './',
    'model':
        {
            'learn_on_entire_history': False,
            'use_sensitve_attributes': False,
            'bias': True,
            'initial_theta': [0.0, 0.0]
        },
    'parameter_optimization': {
        'time_steps': 200,
        'epochs': 1,
        'batch_size': 256,
        'learning_rate': 1,
        'decay_rate': 1,
        'decay_step': 10000,
        'num_decisions': 128 * 256
    },
    'data': {
        'distribution': FICODistribution(bias=True, fraction_protected=0.5),
        'num_test_samples': 8192
    }
}

# Benefit different for demographic parity and equality of opportunity as well as covariance for DP
fairness_functions = [(lambda **fairness_params: fairness_function(type="BD_DP", **fairness_params),
                       lambda **fairness_params: fairness_function_gradient(type="BD_DP", **fairness_params)),
                      (lambda **fairness_params: fairness_function(type="COV_DP", **fairness_params),
                       lambda **fairness_params: fairness_function_gradient(type="COV_DP", **fairness_params)),
                      (lambda **fairness_params: fairness_function(type="BD_EOP", **fairness_params),
                       lambda **fairness_params: fairness_function_gradient(type="BD_EOP", **fairness_params))]

# FICO, COMPAS and AdultCredit
distributions = [FICODistribution(0.5, bias=True),
                 COMPASDistribution(test_percentage=0.2, bias=True),
                 AdultCreditDistribution(test_percentage=0.2, bias=True)]

# Utility for cost 0.1, ..., 0.9
utility_functions = [lambda **util_params: cost_utility(cost_factor=c, **util_params) for c in
                     [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]

for distribution in distributions:
    training_parameters['data']['distribution'] = distribution
    training_parameters['model']['feature_map'] = IdentityFeatureMap(
        training_parameters['distribution'].feature_dimension)

    for fairness_fct, fairness_grad_fct in fairness_functions:
        training_parameters['model']['fairness_gradient_function'] = fairness_grad_fct
        training_parameters['model']['fairness_function'] = fairness_fct

        for utility_function in utility_functions:
            training_parameters['model']['utility_function'] = utility_function

            training_parameters["experiment_name"] = "exp-011-compas-no-fairness"
            training_parameters["model"]["initial_lambda"] = 0.0
