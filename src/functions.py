import os
import sys

root_path = os.path.abspath(os.path.join('.'))
if root_path not in sys.path:
    sys.path.append(root_path)

from src.util import mean

####################### UTILITY FUNCTIONS #######################

def _utility_core(cost_factor, **utility_parameters):
    decisions = utility_parameters["decisions"]
    y = utility_parameters["y"]

    utility = decisions * (y - cost_factor)

    if "ips_weights" in utility_parameters and utility_parameters["ips_weights"] is not None:
        utility *= utility_parameters["ips_weights"]

    return utility

def cost_utility(cost_factor=0.5, **utility_parameters):
    utility = _utility_core(cost_factor, **utility_parameters)
    return mean(utility, axis=0)

def cost_utility_gradient(cost_factor=0.5, **utility_parameters):
    utility = _utility_core(cost_factor=cost_factor, **utility_parameters)
    log_policy_gradient = utility_parameters["policy"].log_policy_gradient(utility_parameters["x"],
                                                                           utility_parameters["s"])
    utility_grad = log_policy_gradient * utility
    return mean(utility_grad, axis=0)

def cost_utility_probability(cost_factor=0.5, **utility_parameters):
    decision_probabilities = utility_parameters["decision_probabilities"]
    y = utility_parameters["y"]

    return mean(decision_probabilities * (y - cost_factor), axis=0)

