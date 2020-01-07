import sys
import os

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import numpy as np
from src.policy import LogisticPolicy
from src.distribution import SplitDistribution

def collect_data(pi, gt_dist, num_samples, fraction_protected):
    x, s, y = collect_unbiased_data(gt_dist, num_samples, fraction_protected)

    decisions = pi(x, s)
    pos_decision_idx = np.arange(x.shape[0])
    pos_decision_idx = pos_decision_idx[decisions == 1]

    return x[pos_decision_idx], s[pos_decision_idx], y[pos_decision_idx]

def collect_unbiased_data(gt_dist, num_samples, fraction_protected):
    x, s = gt_dist.sample_features(num_samples, fraction_protected)
    y = gt_dist.sample_labels(x, s)

    return x, s, y

def train(**training_args):
    gt_dist = SplitDistribution(bias=training_args["bias"])
    learning_parameters = training_args["learning_parameters"]

    pi = LogisticPolicy(
        training_args["dim_theta"], 
        training_args["fairness_function"], 
        training_args["benefit_value_function"], 
        training_args["utility_value_function"], 
        training_args["feature_map"], 
        training_args["fairness_rate"], 
        training_args["use_sensitve_attributes"],
        training_args["keep_collected_data"])

    learning_rate = learning_parameters["learning_rate"]
    x_test, s_test, y_test = collect_unbiased_data(gt_dist, training_args["num_test_samples"], training_args["fraction_protected"])

    # Initial data collection
    x, s, y = collect_data(pi, gt_dist, training_args["num_decisions"], training_args["fraction_protected"])

    for i in range(1, training_args["time_steps"] + 1):        
        if i % learning_parameters['decay_step'] == 0:
            learning_rate *= learning_parameters['decay_rate']
        # train the policy
        pi.update(x, s, y, learning_rate, training_args["batch_size"], training_args["num_iterations"])

        # Collect new data
        x, s, y = collect_data(pi, gt_dist, training_args["num_decisions"], training_args["fraction_protected"])

        # evaluate the policy performance
        decisions_test = pi(x_test, s_test)
        regularized_utility = pi.regularized_utility(x_test, s_test, y_test, decisions_test)
        utility = pi.utility(x_test, s_test, y_test, decisions_test)
        benefit_delta = pi.benefit_delta(x_test, s_test, y_test, decisions_test)

        yield regularized_utility, utility, benefit_delta