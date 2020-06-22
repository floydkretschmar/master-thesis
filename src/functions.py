import os
import sys

root_path = os.path.abspath(os.path.join('.'))
if root_path not in sys.path:
    sys.path.append(root_path)

import numpy as np
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


####################### OPTIMIZER FUNCTIONS #######################

def initialize_adam():
    return {
        "beta_1": 0.9,
        "beta_2": 0.999,
        "v": 0,
        "m": 0,
        "t": 1,
        "epsilon": 0.0000001
    }

def adam_update(update_parameters, gradient):
    update_parameters["m"] = update_parameters["beta_1"] * update_parameters["m"] + (
                1 - update_parameters["beta_1"]) * gradient
    update_parameters["v"] = update_parameters["beta_2"] * update_parameters["v"] + (
                1 - update_parameters["beta_2"]) * np.power(gradient, 2)
    m_hat = update_parameters["m"] / (1 - np.power(update_parameters["beta_1"], update_parameters["t"]))
    v_hat = update_parameters["v"] / (1 - np.power(update_parameters["beta_2"], update_parameters["t"]))
    update_parameters["t"] += 1
    return m_hat / (np.sqrt(v_hat) + update_parameters["epsilon"])

def initialize_sgd():
    return { }

def sgd_update(update_parameters, gradient):
    return gradient