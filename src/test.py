import os
import sys
root_path = os.path.abspath(os.path.join('.'))
if root_path not in sys.path:
    sys.path.append(root_path)

import numpy as np
import multiprocessing as mp

from src.feature_map import IdentityFeatureMap
from src.functions import cost_utility, demographic_parity
from src.plotting import plot_median, plot_mean
from src.training import train
from src.distribution import SplitDistribution, UncalibratedScore

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

def fairness_function(**fairness_kwargs):
    policy = fairness_kwargs["policy"]
    x = fairness_kwargs["x"]
    s = fairness_kwargs["s"]
    y = fairness_kwargs["y"]
    decisions = fairness_kwargs["decisions"]
    ips_weights = fairness_kwargs["ips_weights"]
    numerator, denominator_log_grad = policy._log_gradient(x, s)

    benefit = policy.benefit_function(decisions=decisions, y=y)
    benefit_gradient = benefit / denominator_log_grad

    if ips_weights is not None:
        benefit *= ips_weights
        benefit_gradient *= ips_weights

    benefit_grad = numerator * benefit_gradient
        
    # benefit-difference * grad-benefit-difference
    return policy._mean_difference(benefit, s), policy._mean_difference(benefit_grad, s)

i = 1
bias = True
dim_x = 1
dim_theta = dim_x + 1 if bias else dim_x

def util_func(**util_params):
    util = cost_utility(cost_factor=0.142, **util_params)
    return util

training_parameters = {    
    'save_path': './',
    "experiment_name": "test",
    'model':{
        'benefit_function': demographic_parity,
        'utility_function': util_func,
        'fairness_function': fairness_function,
        'feature_map': IdentityFeatureMap(dim_theta),
        'learn_on_entire_history': False,
        'use_sensitve_attributes': False,
        'bias': bias,
        'initial_theta': [-3.0, 5.0]
    },
    'parameter_optimization': {
        'time_steps':200,
        'epochs': 1,
        'batch_size':256,
        'learning_rate': 1,
        'decay_rate': 1,
        'decay_step': 10000,
        'num_decisions': 128 * 256
    },
    'data': {
        'distribution': UncalibratedScore(bias=bias),
        'fraction_protected':0.5,
        'num_test_samples': 8192
    }
}
lambdas = np.logspace(-2, 1, base=10, endpoint=True, num=20)
lambdas = np.insert(arr=lambdas, obj=0, values=[0.0])
training_parameters["model"]["initial_lambda"] = lambdas

# lambdas = np.logspace(-2, 1, base=10, endpoint=True, num=19)
# lambdas = np.insert(arr=lambdas, obj=0, values=[0.0])
# training_parameters["optimization"]["parameters"]["lambda"] = lambdas

#training_parameters["save_path"] = "/home/fkretschmar/Documents/master-thesis/res/test/uncalibrated/time"
#lambdas = np.logspace(-1, 1, base=10, endpoint=True, num=3)
#lambdas = np.insert(arr=lambdas, obj=0, values=[0.0])

statistics, model_parameters, run_path = train(training_parameters, iterations=10, asynchronous=True)
#statistics, run_path = train(training_parameters, fairness_rates=lambdas, iterations=5, verbose=True, asynchronous=False)
#statistics, run_path = train(training_parameters, fairness_rates=[0.0], iterations=5, verbose=True, asynchronous=False)

#plot_median(statistics, model_parameters=model_parameters)
plot_median(statistics, model_parameters=None)
#plot_mean(statistics, "{}/results_mean_lambdas.png".format(run_path))

#plot_median_over_lambdas(statistics, "{}/results_median_lambdas.png".format(run_path))

#plot_median_over_time(statistics, "{}/results_median.png".format(run_path))