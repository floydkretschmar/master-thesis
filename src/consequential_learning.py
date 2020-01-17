import sys
import os

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import numpy as np
from src.policy import LogisticPolicy
from src.util import save_dictionary, load_dictionary

def apply_policy(data, pi):
    x, s, y = data

    decisions = pi(x, s)
    pos_decision_idx = np.expand_dims(np.arange(decisions.shape[0]), axis=1)
    pos_decision_idx = pos_decision_idx[decisions == 1]

    return x[pos_decision_idx], s[pos_decision_idx], y[pos_decision_idx]

def consequential_learning(**training_args):
    pi = LogisticPolicy(
        training_args["model"]["theta"], 
        training_args["model"]["fairness_function"], 
        training_args["model"]["benefit_value_function"], 
        training_args["model"]["utility_value_function"], 
        training_args["model"]["feature_map"], 
        training_args["optimization"]["fairness_rate"], 
        training_args["model"]["use_sensitve_attributes"],
        training_args["model"]["keep_collected_data"])

    learning_rate = training_args["optimization"]["learning_rate"]

    # Collect test data
    if training_args["data"]["keep_data_across_lambdas"]:
        x_test, s_test, y_test = training_args["data"]["test_dataset"]
    else:
        distribution = training_args["data"]["distribution"]
        x_test, s_test, y_test = distribution.sample_dataset(training_args["data"]["num_test_samples"], training_args["data"]["fraction_protected"])

    for i in range(0, training_args["optimization"]["time_steps"]): 
        # decay learning rate 
        if i % training_args["optimization"]['decay_step'] == 0 and i != 0:
            learning_rate *= training_args["optimization"]['decay_rate']    

        # Collect training data
        if training_args["data"]["keep_data_across_lambdas"]:
            data = training_args["data"]["training_datasets"][i]
        else:
            data = distribution.sample_dataset(training_args["data"]["num_decisions"], training_args["data"]["fraction_protected"])
        
        x, s, y = apply_policy(data, pi)

        # train the policy
        pi.update(x, s, y, learning_rate, training_args["optimization"]["batch_size"], training_args["optimization"]["epochs"])

        # evaluate the policy performance
        decisions_test = pi(x_test, s_test)
        # regularized_utility = pi.regularized_utility(x_test, s_test, y_test, decisions_test)
        utility = pi.utility(x_test, s_test, y_test, decisions_test)
        benefit_delta = pi.benefit_delta(x_test, s_test, y_test, decisions_test)

        yield utility, benefit_delta, pi
