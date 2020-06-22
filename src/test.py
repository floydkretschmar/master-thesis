import os
import sys

import numpy as np
import torch

module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)

from src.functions import cost_utility, cost_utility_gradient, cost_utility_probability
from src.plotting import plot_median
from src.training import train
from src.training_evaluation import UTILITY, COVARIANCE_OF_DECISION_DP
from src.feature_map import IdentityFeatureMap
from src.policy import LogisticPolicy, NeuralNetworkPolicy
from src.distribution import COMPASDistribution, FICODistribution
from src.optimization import ManualGradientLagrangianOptimizationTarget, LagrangianOptimizationTarget, PenaltyOptimizationTarget
from src.util import mean, mean_difference, get_list_of_seeds


# region Fairness Definitions

def calc_benefit(decisions, ips_weights):
    if ips_weights is not None:
        decisions = decisions * ips_weights

    return decisions


def calc_covariance(s, decisions, ips_weights):
    new_s = 1 - (2 * s)

    if ips_weights is not None:
        mu_s = (new_s * ips_weights).mean(0)
        d = decisions * ips_weights
    else:
        mu_s = new_s.mean(0)
        d = decisions

    covariance = (new_s - mu_s) * d
    return covariance


def fairness_function_gradient(type, **fairness_kwargs):
    policy = fairness_kwargs["policy"]
    x = fairness_kwargs["x"]
    s = fairness_kwargs["s"]
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
        return grad.mean(0)


def fairness_function(type, nn, **fairness_kwargs):
    s = fairness_kwargs["s"]
    if nn:
        decisions = fairness_kwargs["decision_probabilities"]
    else:
        decisions = fairness_kwargs["decisions"]

    ips_weights = fairness_kwargs["ips_weights"] if "ips_weights" in fairness_kwargs else None

    if type == "BD_DP":
        benefit = calc_benefit(decisions, ips_weights)
        return mean_difference(benefit, s)
    elif type == "COV_DP":
        covariance = calc_covariance(s, decisions, ips_weights)
        return covariance.mean(0)

def no_fairness(**fairness_kwargs):
    return 0.0

def eval_covariance_of_decision(s, y, decisions):
    return fairness_function(
        type="COV_DP",
        nn=False,
        x=None,
        s=s,
        y=y,
        decisions=decisions,
        ips_weights=None,
        policy=None)


# endregion

def utility(**util_params):
    return cost_utility(cost_factor=0.5, **util_params)

def utility_gradient(**util_params):
    return cost_utility_gradient(cost_factor=0.5, **util_params)

def utility_nn(**util_params):
    return cost_utility_probability(cost_factor=0.5, **util_params)

def covariance_of_decision(**fairness_params):
    return fairness_function(
        type="COV_DP",
        nn=False,
        **fairness_params)

def benefit_difference_dp(**fairness_params):
    return fairness_function(
        type="BD_DP",
        nn=False,
        **fairness_params)

def benefit_difference_dp_nn(**fairness_params):
    return fairness_function(
        type="BD_DP",
        nn=True,
        **fairness_params)

def covariance_of_decision_grad(**fairness_params):
    return fairness_function_gradient(
        type="COV_DP",
        **fairness_params)

def benefit_difference_dp_grad(**fairness_params):
    return fairness_function_gradient(
        type="BD_DP",
        **fairness_params)

def benefit_difference_eop_grad(**fairness_params):
    return fairness_function_gradient(
        type="BD_EOP",
        **fairness_params)


bias = True
distribution = FICODistribution(bias=bias, fraction_protected=0.5)
dim_theta = distribution.feature_dimension

# optim_target = ManualGradientLagrangianOptimizationTarget(0.0,
#                                                            utility,
#                                                            utility_gradient,
#                                                            benefit_difference_dp,
#                                                            benefit_difference_dp_grad)
# optim_target = LagrangianOptimizationTarget(0.0, utility, benefit_difference_dp)
optim_target = PenaltyOptimizationTarget(0.0, utility_nn, benefit_difference_dp_nn)

training_parameters = {
    'model': NeuralNetworkPolicy(distribution.feature_dimension, False),
    #'model': LogisticPolicy(IdentityFeatureMap(dim_theta), False),
    'distribution': distribution,
    'optimization_target': optim_target,
    'parameter_optimization': {
        'time_steps': 10,
        'epochs': 200,
        'batch_size': 128,
        'learning_rate': 0.1,
        'learn_on_entire_history': True,
        'change_iterations': 5
    },
    'data': {
        'num_train_samples': 256 * 30,
        'num_test_samples': 1024
    },
    # 'lagrangian_optimization': {
    #     'epochs': 200,
    #     'batch_size': 4096,
    #     'learning_rate': 0.01,
    # },
    'evaluation': {
        UTILITY: {
            'measure_function': utility,
            'detailed': False
        },
        COVARIANCE_OF_DECISION_DP: {
            'measure_function': eval_covariance_of_decision,
            'detailed': False
        }
    }
}


training_parameters["save_path"] = "../res/local_experiments/TEST"
statistics, model_parameters, run_path = train(
    training_parameters,
    iterations=[200],
    asynchronous=False,
    fairness_rates=[0.0])

plot_median(x_values=range(training_parameters["parameter_optimization"]["time_steps"] + 1),
            x_label="Time steps",
            x_scale="linear",
            performance_measures=[statistics.get_additonal_measure(UTILITY, "Utility"),
                                  statistics.demographic_parity(),
                                  statistics.equality_of_opportunity()],
            fairness_measures=[],
            file_path="{}/results_median_time.png".format(run_path))
