import os
import sys
root_path = os.path.abspath(os.path.join('.'))
if root_path not in sys.path:
    sys.path.append(root_path)

import numpy as np

from src.feature_map import IdentityFeatureMap
from src.functions import cost_utility, demographic_parity
from src.plotting import plot_median
from src.training import train
from src.distribution import ResamplingDistribution
from src.util import load_dataset


# def fairness_function(**fairness_kwargs):
#     policy = fairness_kwargs["policy"]
#     x = fairness_kwargs["x"]
#     s = fairness_kwargs["s"]
#     y = fairness_kwargs["y"]
#     decisions = fairness_kwargs["decisions"]
#     ips_weights = fairness_kwargs["ips_weights"]
#     numerator, denominator_log_grad = policy._log_gradient(x, s)

#     benefit = policy.benefit_function(decisions=decisions, y=y)

#     if ips_weights is not None:
#         mu_s = np.mean(s * ips_weights, axis=0)
#     else:
#         mu_s = np.mean(s, axis=0)

#     if ips_weights is not None:
#         benefit *= ips_weights

#     covariance = (s - mu_s) * benefit
#     covariance_grad = (covariance * numerator)/denominator_log_grad

#     return np.mean(covariance, axis=0), np.mean(covariance_grad, axis=0)

def calc_benefit(**fairness_kwargs):
    y = fairness_kwargs["y"]
    decisions = fairness_kwargs["decisions"]
    ips_weights = fairness_kwargs["ips_weights"]

    benefit = demographic_parity(decisions=decisions, y=y)

    if ips_weights is not None:
        benefit *= ips_weights

    return benefit


def fairness_gradient_function(**fairness_kwargs):
    policy = fairness_kwargs["policy"]
    x = fairness_kwargs["x"]
    s = fairness_kwargs["s"]
    benefit = calc_benefit(**fairness_kwargs)

    log_gradient = policy.log_policy_gradient(x, s)
    benefit_grad = log_gradient * benefit

    return policy._mean_difference(benefit_grad, s)


def fairness_function(**fairness_kwargs):
    policy = fairness_kwargs["policy"]
    s = fairness_kwargs["s"]
    benefit = calc_benefit(**fairness_kwargs)

    return policy._mean_difference(benefit, s)


bias = True
distribution = ResamplingDistribution(bias=bias, dataset=load_dataset("./dat/compas/compas.npz"), test_percentage=0.2)
dim_theta = distribution.feature_dim

print(dim_theta)


def util_func(**util_params):
    util = cost_utility(cost_factor=0.6, **util_params)
    return util


training_parameters = {
    'save_path': './',
    'model': {
        'benefit_function': demographic_parity,
        'utility_function': util_func,
        'fairness_function': fairness_function,
        'fairness_gradient_function': fairness_gradient_function,
        'feature_map': IdentityFeatureMap(dim_theta),
        'learn_on_entire_history': False,
        'use_sensitve_attributes': False,
        'bias': bias,
        'initial_theta': np.zeros((dim_theta))
    },
    'parameter_optimization': {
        'time_steps': 200,
        'epochs': 40,
        'batch_size': 64,
        'learning_rate': 0.1,
        'decay_rate': 1,
        'decay_step': 10000,
        'num_decisions': 4096
    },
    'data': {
        'distribution': distribution,
        'num_test_samples': None
    }
}

# lambdas = np.logspace(-2, 1, base=10, endpoint=True, num=20)
# #lambdas = np.insert(arr=lambdas, obj=0, values=[0.0])
# training_parameters["model"]["initial_lambda"] = lambdas

training_parameters["experiment_name"] = "Test"
training_parameters["model"]["initial_lambda"] = 0.0
training_parameters["lagrangian_optimization"] = {
    'iterations': 30,
    'epochs': 10,
    'batch_size': 64,
    'learning_rate': 1,
    'decay_rate': 1,
    'decay_step': 10000,
    'num_decisions': 4096
}

# x, s, y = load_dataset("./src/dat/compas/compas.npz")
# x, x_test, y, y_test, s, s_test = train_test_split(x, y, s, test_size=0.8)
# dist = ResamplingDistribution(load_dataset("./dat/compas/compas.npz"), 0.2, bias=bias)

# training_parameters["model"]["initial_lambda"] = 0.0

# lambdas = np.logspace(-2, 1, base=10, endpoint=True, num=19)
# lambdas = np.insert(arr=lambdas, obj=0, values=[0.0])
# training_parameters["optimization"]["parameters"]["lambda"] = lambdas

#training_parameters["save_path"] = "/home/fkretschmar/Documents/master-thesis/res/test/uncalibrated/time"
#lambdas = np.logspace(-1, 1, base=10, endpoint=True, num=3)
#lambdas = np.insert(arr=lambdas, obj=0, values=[0.0])

statistics, model_parameters, run_path = train(training_parameters, iterations=10, asynchronous=False)
#statistics, run_path = train(training_parameters, fairness_rates=lambdas, iterations=5, verbose=True, asynchronous=False)
#statistics, run_path = train(training_parameters, fairness_rates=[0.0], iterations=5, verbose=True, asynchronous=False)

#plot_median(statistics, model_parameters=model_parameters)
plot_median(statistics, model_parameters=None)
#plot_mean(statistics, "{}/results_mean_lambdas.png".format(run_path))

#plot_median_over_lambdas(statistics, "{}/results_median_lambdas.png".format(run_path))

#plot_median_over_time(statistics, "{}/results_median.png".format(run_path))