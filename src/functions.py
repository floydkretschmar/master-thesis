import os
import sys

import numpy as np

root_path = os.path.abspath(os.path.join('.'))
if root_path not in sys.path:
    sys.path.append(root_path)

####################### UTILITY FUNCTIONS #######################

def cost_utility(cost_factor, **utility_parameters):
    decisions = utility_parameters["decisions"]
    y = utility_parameters["y"]

    return decisions * (y - cost_factor)


def log_cost_utility(cost_factor, **utility_parameters):
    # x = utility_parameters["x"]
    # s = utility_parameters["s"]
    y = utility_parameters["y"]
    decision_probabilities = utility_parameters["decision_probabilities"]

    neg_log_likelihood = np.log(decision_probabilities)
    utility = -neg_log_likelihood * (y - cost_factor)

    return utility
