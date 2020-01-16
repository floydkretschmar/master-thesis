import sys
import os
import numpy as np

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from src.feature_map import IdentityFeatureMap
from src.functions import cost_utility, demographic_parity
from src.plotting import plot_results_over_time, plot_results_over_lambdas
from src.training import train_multiple
import multiprocessing as mp

# def fairness_function(**fairness_kwargs):
#     policy = fairness_kwargs["policy"]
#     x = fairness_kwargs["x"]
#     s = fairness_kwargs["s"]
#     y = fairness_kwargs["y"]
#     decisions = fairness_kwargs["decisions"]
#     ips_weights = fairness_kwargs["ips_weights"]

#     phi = policy.feature_map(policy._extract_features(x, s))
#     denominator = np.expand_dims((1.0 + np.exp(np.matmul(phi, policy.theta))), axis=1)
#     benefit = policy.benefit_function(decisions=decisions, y=y)

#     # if ips_weights is not None:
#     #     mu_s = np.mean(s * ips_weights, axis=0)
#     # else:
#     #     mu_s = np.mean(s, axis=0)
#     mu_s = np.mean(s, axis=0)

#     covariance = (s - mu_s) * benefit
#     covariance_grad = covariance * phi

#     if ips_weights is not None:
#         #covariance *= ips_weights
#         covariance_grad *= ips_weights/denominator
#     else:
#         covariance_grad /= denominator

#     return np.mean(covariance, axis=0) * np.mean(covariance_grad, axis=0)

def fairness_function(**fairness_kwargs):
    policy = fairness_kwargs["policy"]
    x = fairness_kwargs["x"]
    s = fairness_kwargs["s"]
    y = fairness_kwargs["y"]
    decisions = fairness_kwargs["decisions"]
    ips_weights = fairness_kwargs["ips_weights"]

    benefit = policy.benefit_function(decisions=decisions, y=y)
    log_gradient = policy._log_gradient(x, s)
    
    benefit_grad = benefit * log_gradient

    if ips_weights is not None:
        benefit *= ips_weights
        benefit_grad *= ips_weights
        
    # benefit-difference * grad-benefit-difference
    return policy._mean_difference(benefit, s) * policy._mean_difference(benefit_grad, s)

i = 1
dim_x = 1
training_parameters = {
    'keep_collected_data': False,
    'use_sensitve_attributes': False,
    'time_steps':200,
    'batch_size':512,
    'num_iterations': 32,
    'learning_parameters': {
        'learning_rate': 0.5,
        'decay_rate': 0.8,
        'decay_step': 30
    },
    'fraction_protected':0.5,
    'num_test_samples': 2000,
    'bias': True,
    'benefit_value_function': demographic_parity
}
def util_func(**util_params):
    util = cost_utility(cost_factor=0.55, **util_params)
    return util

dim_theta = dim_x + 1 if training_parameters['bias'] else dim_x
training_parameters['theta'] = [-3.5, 0.6]
training_parameters['feature_map'] = IdentityFeatureMap(dim_theta)
training_parameters['num_decisions'] = training_parameters['num_iterations'] * training_parameters['batch_size']
training_parameters['utility_value_function'] = util_func
training_parameters['fairness_function'] = fairness_function

lambdas = np.logspace(-1, 5, base=10, endpoint=True, num=10)
#lambdas = np.insert(arr=lambdas, obj=0, values=[0.0])

results = train_multiple(training_parameters, iterations=5, lambdas=[100.0], verbose=True, asynchronous=False)