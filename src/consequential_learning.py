import sys
import os

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import numpy as np
from src.policy import LogisticPolicy
from src.util import save_dictionary, load_dictionary

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
        training_args["model"]["use_sensitve_attributes"],
        training_args["model"]["keep_collected_data"])

    learning_rate = training_args["optimization"]["learning_rate"]

    for i in range(0, training_args["optimization"]["time_steps"]): 
        yield pi

        # decay learning rate 
        if i % training_args["optimization"]['decay_step'] == 0 and i != 0:
            learning_rate *= training_args["optimization"]['decay_rate']    

        # Collect training data
        data = training_args["data"]["training_datasets"][i]
        
        x, s, y = apply_policy(data, pi)

        # train the policy
        pi.update(x, s, y, learning_rate, training_args["optimization"]["batch_size"], training_args["optimization"]["epochs"])
    
    yield pi