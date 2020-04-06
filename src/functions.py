import os
import sys
from copy import deepcopy

import numpy as np

root_path = os.path.abspath(os.path.join('.'))
if root_path not in sys.path:
    sys.path.append(root_path)


####################### BENEFIT FUNCTIONS #######################
def demographic_parity(**benefit_parameters):
    decisions = deepcopy(benefit_parameters["decisions"])
    return decisions


def equal_opportunity(**benefit_parameters):
    y_s = benefit_parameters["y"]
    decisions = benefit_parameters["decisions"]
    return y_s * decisions


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
