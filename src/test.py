import sys
import os
import numpy as np

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from src.feature_map import IdentityFeatureMap
from src.functions import cost_utility, demographic_parity
from src.plotting import plot_median_over_time
from src.training import train
import multiprocessing as mp
from src.distribution import SplitDistribution, UncalibratedScore

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
bias = True
dim_x = 1
dim_theta = dim_x + 1 if bias else dim_x

def util_func(**util_params):
    util = cost_utility(cost_factor=0.1, **util_params)
    return util

training_parameters = {    
    'save_path': "/home/fkretschmar/Documents/master-thesis/res/exp-006/uncalibrated",
    'model':{
        'theta': [-3.0, 5.0],
        'benefit_value_function': demographic_parity,
        'utility_value_function': util_func,
        'fairness_function': fairness_function,
        'feature_map': IdentityFeatureMap(dim_theta),
        'keep_collected_data': False,
        'use_sensitve_attributes': False,
        'bias': bias
    },
    'optimization': {
        'time_steps':5,
        'epochs': 128,
        'batch_size':256,
        'learning_rate': 1,
        'decay_rate': 1,
        'decay_step': 10000,
        'test_at_every_timestep': False
    },
    'data': {
        'distribution': UncalibratedScore(bias=bias),
        'keep_data_across_lambdas': True,
        'fraction_protected':0.5,
        'num_test_samples': 8192,
        'num_decisions': 128 * 256
    }
}


statistics, run_path = train(training_parameters, fairness_rates=[0.0], iterations=2, store_all=True, verbose=True, asynchronous=False)

plot_median_over_time(statistics, "{}/results_median.png".format(run_path))