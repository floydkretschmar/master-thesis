import os
import sys

import numpy as np

module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)

from src.util import mean_difference
from src.feature_map import IdentityFeatureMap
from src.functions import cost_utility
from src.plotting import plot_mean, plot_median
from src.training import train
from src.training_evaluation import UTILITY, COVARIANCE_OF_DECISION_DP
from src.policy import LogisticPolicy
from src.distribution import COMPASDistribution
from src.optimization import PenaltyOptimizationTarget


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
    elif type == "COV_DP_DIST":
        phi = policy.feature_map(policy._extract_features(x, s))
        distance = np.matmul(phi, policy.theta).reshape(-1, 1)
        result = calc_covariance(s, distance, ips_weights)

    log_gradient = policy.log_policy_gradient(x, s)
    grad = log_gradient * result

    if type == "BD_DP":
        return mean_difference(grad, s)
    elif type == "COV_DP":
        return np.mean(grad, axis=0)
    elif type == "COV_DP_DIST":
        return np.mean(grad, axis=0)
    elif type == "BD_EOP":
        y1_indices = np.where(y == 1)
        return mean_difference(grad[y1_indices], s[y1_indices])


def fairness_function(type, **fairness_kwargs):
    x = fairness_kwargs["x"]
    s = fairness_kwargs["s"]
    y = fairness_kwargs["y"]
    decisions = fairness_kwargs["decisions"]
    ips_weights = fairness_kwargs["ips_weights"]
    policy = fairness_kwargs["policy"]

    if type == "BD_DP":
        benefit = calc_benefit(decisions, ips_weights)
        return mean_difference(benefit, s)
    elif type == "COV_DP":
        covariance = calc_covariance(s, decisions, ips_weights)
        return np.mean(covariance, axis=0)
    elif type == "COV_DP_DIST":
        phi = policy.feature_map(policy._extract_features(x, s))
        distance = np.matmul(phi, policy.theta).reshape(-1, 1)
        covariance = calc_covariance(s, distance, ips_weights)
        return np.mean(covariance, axis=0)
    elif type == "BD_EOP":
        benefit = calc_benefit(decisions, ips_weights)
        y1_indices = np.where(y == 1)
        return mean_difference(benefit[y1_indices], s[y1_indices])


bias = True
distribution = COMPASDistribution(bias=bias, test_percentage=0.2)
dim_theta = distribution.feature_dimension


def util_func(**util_params):
    util = cost_utility(cost_factor=0.5, **util_params)
    return util


training_parameters = {
    'model': {
        'constructor': LogisticPolicy,
        'parameters': {
            "theta": np.zeros((dim_theta)),
            "feature_map": IdentityFeatureMap(dim_theta),
            "use_sensitive_attributes": False
        }
    },
    'distribution': distribution,
    'optimization_target': {
        'constructor': PenaltyOptimizationTarget,
        'parameters': {
            'utility_function': util_func
        }
    },
    'parameter_optimization': {
        'time_steps': 200,
        'epochs': 150,
        'batch_size': 128,
        'learning_rate': 0.01,
        'learn_on_entire_history': False,
        'fix_seeds': True,
        'standardize_ips_weights': True
    },
    'data': {
        'num_train_samples': 4096,
        'num_test_samples': 1024
    },
    'evaluation': {
        UTILITY: {
            'measure_function': lambda s, y, decisions: np.mean(util_func(s=s,
                                                                          y=y,
                                                                          decisions=decisions)),
            'detailed': False
        },
        COVARIANCE_OF_DECISION_DP: {
            'measure_function': lambda s, y, decisions: fairness_function(
                type="COV_DP",
                x=None,
                s=s,
                y=y,
                decisions=decisions,
                ips_weights=None,
                policy=None),
            'detailed': False
        }
    }
}

training_parameters['optimization_target']['parameters']['fairness_function'] \
    = lambda **fp: fairness_function("BD_DP", **fp)
training_parameters['optimization_target']['parameters']['fairness_gradient_function'] \
    = lambda **fp: fairness_function_gradient("BD_DP", **fp)

training_parameters["save_path"] = "../res/local_experiments/NO_FAIRNESS"
statistics, model_parameters, run_path = train(
    training_parameters,
    iterations=1,
    asynchronous=False,
    fairness_rates=[0.0])

plot_mean(x_values=range(training_parameters["parameter_optimization"]["time_steps"] + 1),
          x_label="Time steps",
          x_scale="linear",
          performance_measures=[statistics.get_additonal_measure(UTILITY, "Utility"),
                                statistics.accuracy()],
          fairness_measures=[statistics.demographic_parity(),
                             statistics.equality_of_opportunity(),
                             statistics.get_additonal_measure(COVARIANCE_OF_DECISION_DP,
                                                              "Covariance of Decision (DP)")],
          file_path="{}/results_mean_time.png".format(run_path))
plot_median(x_values=range(training_parameters["parameter_optimization"]["time_steps"] + 1),
            x_label="Time steps",
            x_scale="linear",
            performance_measures=[statistics.get_additonal_measure(UTILITY, "Utility"),
                                  statistics.accuracy()],
            fairness_measures=[statistics.demographic_parity(),
                               statistics.equality_of_opportunity(),
                               statistics.get_additonal_measure(COVARIANCE_OF_DECISION_DP,
                                                                "Covariance of Decision (DP)")],
            file_path="{}/results_median_time.png".format(run_path))
