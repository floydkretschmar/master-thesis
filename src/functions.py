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

def cost_utility_probability(cost_factor, **utility_parameters):
    decisions = utility_parameters["decision_probabilities"]
    y = utility_parameters["y"]

    return decisions * (y - cost_factor)
