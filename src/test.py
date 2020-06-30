import os
import sys

import numpy as np
import torch
from pathlib import Path

module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)

from src.functions import cost_utility, cost_utility_gradient, cost_utility_probability, SGD, ADAM
from src.plotting import plot_median, Plot
from src.training import train
from src.training_evaluation import UTILITY, COVARIANCE_OF_DECISION_DP
from src.feature_map import IdentityFeatureMap
from src.policy import LogisticPolicy, NeuralNetworkPolicy
from src.distribution import COMPASDistribution, FICODistribution, AdultCreditDistribution
from src.optimization import ManualGradientLagrangianOptimizationTarget, LagrangianOptimizationTarget, \
    PenaltyOptimizationTarget, ManualGradientPenaltyOptimizationTarget, ManualGradientAugmentedLagrangianOptimizationTarget, AugmentedLagrangianOptimizationTarget
from src.util import mean, mean_difference, get_list_of_seeds, fix_seed

# region Fairness Definitions
fix_seed(3403344728)


def calc_benefit(decisions, ips_weights):
    if ips_weights is not None:
        decisions = decisions * ips_weights

    return decisions


def calc_covariance(s, decisions, ips_weights):
    # change label s in {0, 1} to s in {-1, 1}
    # new_s = (2 * s) - 1
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


def covariance_of_decision_nn(**fairness_params):
    return fairness_function(
        type="COV_DP",
        nn=True,
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
# distribution = COMPASDistribution(bias=bias, test_percentage=0.2)
# distribution = AdultCreditDistribution(bias=bias, test_percentage=0.2)
dim_theta = distribution.feature_dimension
fairness_lr = 0.01

# optim_target = ManualGradientLagrangianOptimizationTarget(0.0,
#                                                           utility,
#                                                           utility_gradient,
#                                                           benefit_difference_dp,
#                                                           benefit_difference_dp_grad,
#                                                           # covariance_of_decision,
#                                                           # covariance_of_decision_grad,
#                                                           # error_delta=0.001)
#                                                           error_delta=0.0)

# optim_target = ManualGradientAugmentedLagrangianOptimizationTarget(0.0,
#                                                           utility,
#                                                           utility_gradient,
#                                                           # benefit_difference_dp,
#                                                           # benefit_difference_dp_grad,
#                                                           covariance_of_decision,
#                                                           covariance_of_decision_grad,
#                                                           penalty_constant=fairness_lr)
# optim_target = ManualGradientPenaltyOptimizationTarget(0.0,
#                                                        utility,
#                                                        utility_gradient,
#                                                        benefit_difference_dp,
#                                                        benefit_difference_dp_grad)
optim_target = AugmentedLagrangianOptimizationTarget(0.0, utility_nn, covariance_of_decision_nn, penalty_constant=fairness_lr)
# optim_target = LagrangianOptimizationTarget(0.0, utility_nn, covariance_of_decision_nn, error_delta=0.0)
# optim_target = PenaltyOptimizationTarget(0.0, utility_nn, benefit_difference_dp_nn)


training_parameters = {
    'model': NeuralNetworkPolicy(distribution.feature_dimension, False),
    # 'model': LogisticPolicy(IdentityFeatureMap(dim_theta), False),
    'distribution': distribution,
    'optimization_target': optim_target,
    'parameter_optimization': {
        'time_steps': 50,
        # 'time_steps': 1,
        'epochs': 50,
        'batch_size': 64,
        'learning_rate': 0.001,
        # 'learning_rate': 0.1,
        'learn_on_entire_history': True,
        'clip_weights': True,
        # 'training_algorithm': ADAM,
        'training_algorithm': torch.optim.Adam,
    },
    'data': {
        'num_train_samples': 128,
        # 'num_train_samples': 4096,
        'num_test_samples': 1600
    },
    'lagrangian_optimization': {
        # 'epochs': 200,
        'epochs': 10,
        'batch_size': 6400,
        'learning_rate': fairness_lr,
        'training_algorithm': ADAM,
    },
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

# training_parameters = {
#     'model': NeuralNetworkPolicy(distribution.feature_dimension, False),
#     # 'model': LogisticPolicy(IdentityFeatureMap(dim_theta), False),
#     'distribution': distribution,
#     'optimization_target': optim_target,
#     'parameter_optimization': {
#         'time_steps': 50,
#         'epochs': 50,
#         'batch_size': 98,
#         'learning_rate': 0.001,
#         # 'learning_rate': 0.1,
#         'learn_on_entire_history': True,
#         'clip_weights': True,
#         'training_algorithm': torch.optim.Adam
#     },
#     'data': {
#         'num_train_samples': 98,
#         'num_test_samples': 1235
#     },
#     'lagrangian_optimization': {
#         'epochs': 30,
#         'batch_size': 4900,
#         'training_algorithm': ADAM,
#         # 'batch_size': 98,
#         'learning_rate': fairness_lr,
#     },
#     'evaluation': {
#         UTILITY: {
#             'measure_function': utility,
#             'detailed': False
#         },
#         COVARIANCE_OF_DECISION_DP: {
#             'measure_function': eval_covariance_of_decision,
#             'detailed': False
#         }
#     }
# }

# training_parameters = {
#     'model': NeuralNetworkPolicy(distribution.feature_dimension, False),
#     # 'model': LogisticPolicy(IdentityFeatureMap(dim_theta), False),
#     'distribution': distribution,
#     'optimization_target': optim_target,
#     'parameter_optimization': {
#         'time_steps': 50,
#         # 'time_steps': 1,
#         'epochs': 50,
#         'batch_size': 781,
#         'learning_rate': 0.001,
#         # 'learning_rate': 0.1,
#         'learn_on_entire_history': True,
#         'clip_weights': True,
#         'training_algorithm': torch.optim.Adam
#     },
#     'data': {
#         'num_train_samples': 256,
#         # 'num_train_samples': 4096,
#         'num_test_samples': 9792
#     },
#     'lagrangian_optimization': {
#         # 'epochs': 200,
#         'epochs': 10,
#         'batch_size': 50000,
#         'learning_rate': fairness_lr,
#         'training_algorithm': ADAM,
#         # 'decay_step': 10,
#         # 'decay_rate': 0.9
#     },
#     'evaluation': {
#         UTILITY: {
#             'measure_function': utility,
#             'detailed': False
#         },
#         COVARIANCE_OF_DECISION_DP: {
#             'measure_function': eval_covariance_of_decision,
#             'detailed': False
#         }
#     }
# }

def get_plots(statistics, model_parameters):
    plots = []
    plots.append(Plot(range(training_parameters["parameter_optimization"]["time_steps"] + 1),
                      "Time Steps",
                      "linear",
                      "Utility",
                      statistics.get_additonal_measure(UTILITY, "Utility")))
    plots.append(Plot(range(training_parameters["parameter_optimization"]["time_steps"] + 1),
                      "Time Steps",
                      "linear",
                      "Accuracy",
                      statistics.accuracy()))
    plots.append(Plot(range(training_parameters["parameter_optimization"]["time_steps"] + 1),
                      "Time Steps",
                      "linear",
                      statistics.demographic_parity().name,
                      statistics.demographic_parity()))
    plots.append(Plot(range(training_parameters["parameter_optimization"]["time_steps"] + 1),
                      "Time Steps",
                      "linear",
                      statistics.equality_of_opportunity().name,
                      statistics.equality_of_opportunity()))
    plots.append(Plot(range(training_parameters["parameter_optimization"]["time_steps"] + 1),
                      "Time Steps",
                      "linear",
                      "Lagrangian Multipliers",
                      *model_parameters.get_lagrangians()))
    return plots


save_path = '../res/TEST/FICO'.format(fairness_lr)
Path(save_path).mkdir(parents=True, exist_ok=True)

# training_parameters["save_path"] = "../res/local_experiments/TEST"
overall_statistic, overall_model_parameters, _ = train(
    training_parameters,
    fairness_rates=[0.0])
plot_median(performance_plots=get_plots(overall_statistic, overall_model_parameters),
            fairness_plots=[],
            file_path="{}/run_0.png".format(save_path),
            figsize=(20, 10))

for r in range(9):
    statistics, model_parameters, _ = train(
        training_parameters,
        fairness_rates=[0.0])
    plot_median(performance_plots=get_plots(statistics, model_parameters),
                fairness_plots=[],
                file_path="{}/run_{}.png".format(save_path, r + 1),
                figsize=(20, 10))

    overall_statistic.merge(statistics)
    overall_model_parameters.merge(model_parameters)
    dp = overall_statistic.demographic_parity()

plot_median(performance_plots=get_plots(overall_statistic, overall_model_parameters),
            fairness_plots=[],
            file_path="{}/results_median_time.png".format(save_path),
            figsize=(20, 10))
