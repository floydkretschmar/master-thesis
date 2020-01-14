import sys
import os
import numpy as np

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from src.consequential_learning import train
from src.feature_map import IdentityFeatureMap
from src.functions import cost_utility, demographic_parity
from src.plotting import plot_results_over_time

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
    phi = policy.feature_map(policy._extract_features(x, s))
    denominator = np.expand_dims((1.0 + np.exp(np.matmul(phi, policy.theta))), axis=1)

    benefit /= denominator
    benefit_grad = benefit * phi#/denominator

    if ips_weights is not None:
        #benefit *= ips_weights
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
    'fairness_rate':1000,
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
#training_parameters['num_decisions'] = 10000
training_parameters['utility_value_function'] = util_func
training_parameters['fairness_function'] = fairness_function

benefit_deltas = []
utilities = []

for i in range(1, 21):
    benefit_delta = []
    utility = []
    j = 0

    print("Current iteration: {}".format(i))

    for u, bd in train(**training_parameters):
        benefit_delta.append(bd)
        utility.append(u)
        print("Time step {}: Utility {} \n\t Benefit Delta {}".format(j, u, bd))
        j += 1

    benefit_deltas.append(np.array(benefit_delta))
    utilities.append(np.array(utility))

plot_results_over_time(np.array(utilities), np.array(benefit_deltas))

# approx_policy_plus = self.copy()
# approx_policy_plus.theta += epsilon
# approx_policy_minus = self.copy()
# approx_policy_minus.theta -= epsilon
# approx_gradient = (approx_policy_plus.regularized_utility(X_batch[pos_decision_idx], S_batch[pos_decision_idx], Y_batch[pos_decision_idx], sampling_theta) - approx_policy_minus.regularized_utility(X_batch[pos_decision_idx], S_batch[pos_decision_idx], Y_batch[pos_decision_idx], sampling_theta)) / (2 * epsilon)