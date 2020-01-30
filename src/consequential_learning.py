import os
import sys
root_path = os.path.abspath(os.path.join('.'))
if root_path not in sys.path:
    sys.path.append(root_path)

import numpy as np
from copy import deepcopy
from src.policy import LogisticPolicy
from src.util import save_dictionary, load_dictionary
from src.learning_algorithms import StochasticGradientAscent

def apply_policy(data, pi):
    """ Makes decisions for the data based on the specified policy and only returns the x, s and y of the 
    accepted data points, emulating imperfect data collection.
        
    Args:
        data: The dataset of x, s and y based on which the decisions are made.
        pi: The policy used to make the data collection decisions.

    Returns:
        x: The features of the accepted samples
        s: The sensitive attribute of the accepted samples
        y: The ground truth lable of the accepted samples
    """
    x, s, y = data

    decisions = pi(x, s)
    pos_decision_idx = np.expand_dims(np.arange(decisions.shape[0]), axis=1)
    pos_decision_idx = pos_decision_idx[decisions == 1]

    return x[pos_decision_idx], s[pos_decision_idx], y[pos_decision_idx]

def consequential_learning(**training_args):
    """ Executes the consequential learning algorithm according to the specified training parameters.
        
    Args:
        training_args: The parameters used to configure the consequential learning algorithm.

    Returns:
        utility: The utility of the policy on the test data.
        benefit_delta: The benefit delta of the policy on the test data.
    """
    pi = LogisticPolicy(
        training_args["model"]["theta"], 
        training_args["model"]["fairness_function"], 
        training_args["model"]["benefit_function"], 
        training_args["model"]["utility_function"], 
        training_args["model"]["feature_map"], 
        training_args["optimization"]["fairness_rate"], 
        training_args["model"]["use_sensitve_attributes"])

    training_algorithm = StochasticGradientAscent(training_args["model"]["learn_on_entire_history"])
    learning_rates = { key: training_args["optimization"]["learning_rates"][key]["learning_rate"] for key in training_args["optimization"]["learning_rates"] }

    for i in range(0, training_args["optimization"]["time_steps"]): 
        yield pi

        # decay learning rates 
        for parameter in learning_rates:
            if i % training_args["optimization"]["learning_rates"][parameter]['decay_step'] == 0 and i != 0:
                learning_rates[parameter] *= training_args["optimization"]["learning_rates"][parameter]["decay_rate"]

        # Collect training data
        data = training_args["data"]["training_datasets"][i]        
        x, s, y = apply_policy(data, pi)

        # train the policy
        pi = training_algorithm.update(pi, x, s, y, batch_size=training_args["optimization"]["batch_size"], epochs=training_args["optimization"]["epochs"], learning_rates=learning_rates)
        
    yield pi