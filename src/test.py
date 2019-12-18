import sys
import os

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from src.consequential_learning import train
from src.feature_map import IdentityFeatureMap

def fairness_function(**fairness_kwargs):
    return fairness_kwargs['policy'].calculate_benefit_difference(fairness_kwargs['x'], fairness_kwargs['s'], fairness_kwargs['sample_theta'], fairness_kwargs['gradient'], fairness_kwargs['use_s'])
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
    'fairness_rate':0,
    'cost_factor':0.55,
    'fraction_protected':0.3
}
training_parameters['dim_theta'] = training_parameters['dim_x'] + training_parameters['dim_s'] if training_parameters['dim_s'] else training_parameters['dim_x']
training_parameters['feature_map'] = IdentityFeatureMap(training_parameters['dim_theta'])
training_parameters['num_decisions'] = training_parameters['num_iterations'] * training_parameters['batch_size']
utilities = []
benefit_deltas = []

for utility, benefit_delta in train(**training_parameters, fairness_function=fairness_function):
    #print("Time step {}: Utility {} \n\t Benefit Delta {}".format(i, utility, benefit_delta))
    utilities.append(utility)
    benefit_deltas.append(benefit_delta)
    i += 1