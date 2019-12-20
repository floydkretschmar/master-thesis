import sys
import os

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from src.consequential_learning import train
from src.feature_map import IdentityFeatureMap
from src.functions import cost_utility, demographic_parity

def fairness_function(**fairness_kwargs):
    policy = fairness_kwargs["policy"]
    x = fairness_kwargs["x"]
    y = fairness_kwargs["y"]
    s = fairness_kwargs["s"]
    decisions = fairness_kwargs["decisions"].reshape(-1, 1)
    gradient = fairness_kwargs["gradient"]
    sampling_theta = fairness_kwargs["sampling_theta"]

    mu_s = policy._calculate_expectation(x, s, s.astype(float), False, sampling_theta).reshape(-1, 1)
    covariance = (s - mu_s) * decisions

    return policy._calculate_expectation(x, s, covariance, gradient, sampling_theta)

i = 1
training_parameters = {
    'dim_x': 1,
    'dim_s': None,
    'time_steps':200,
    'batch_size':512,
    'num_iterations': 32,
    'learning_parameters': {
        'learning_rate': 0.5,
        'decay_rate': 0.8,
        'decay_step': 30
    },
    'fairness_rate':10,
    'fraction_protected':0.3,
    'num_test_samples': 5000,
    'bias': True,
    'fairness_function': fairness_function,
    'benefit_value_function': demographic_parity
}
training_parameters['dim_x'] = training_parameters['dim_x'] + 1 if training_parameters['bias'] else training_parameters['dim_x']
dim_theta = training_parameters['dim_x'] + training_parameters['dim_s'] if training_parameters['dim_s'] else training_parameters['dim_x']
training_parameters['feature_map'] = IdentityFeatureMap(dim_theta)
training_parameters['num_decisions'] = training_parameters['num_iterations'] * training_parameters['batch_size']
training_parameters['utility_value_function'] = lambda **util_params : cost_utility(cost_factor=0.55, **util_params)

for regularized, utility, benefit_delta in train(**training_parameters):
    print("Time step {}: Utility {} \n\t Benefit Delta {}".format(i, utility, benefit_delta))
    i += 1

# approx_policy_plus = self.copy()
# approx_policy_plus.theta += epsilon
# approx_policy_minus = self.copy()
# approx_policy_minus.theta -= epsilon
# approx_gradient = (approx_policy_plus.regularized_utility(X_batch[pos_decision_idx], S_batch[pos_decision_idx], Y_batch[pos_decision_idx], sampling_theta) - approx_policy_minus.regularized_utility(X_batch[pos_decision_idx], S_batch[pos_decision_idx], Y_batch[pos_decision_idx], sampling_theta)) / (2 * epsilon)