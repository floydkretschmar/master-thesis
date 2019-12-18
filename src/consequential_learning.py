import sys
import os

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import numpy as np
from src.policy import LogisticPolicy
from src.distribution import SplitDistribution

def collect_data(pi, gt_dist, num_samples, fraction_protected):
    x, s = gt_dist.sample_features(num_samples, fraction_protected)
    decisions = pi(x, s)

    pos_decision_idx = np.arange(x.shape[0])
    pos_decision_idx = pos_decision_idx[decisions == 1]

    x = x[pos_decision_idx]
    s = s[pos_decision_idx]

    y = gt_dist.sample_labels(x, s)

    return x, s, y

def train(**training_args):
    gt_dist = SplitDistribution()
    learning_parameters = training_args["learning_parameters"]
    pi = LogisticPolicy(training_args["dim_theta"], training_args["fairness_rate"], training_args["cost_factor"], training_args["fairness_function"], training_args["feature_map"])

    learning_rate = learning_parameters["learning_rate"]

    for i in range(1, training_args["time_steps"] + 1):        
        if i % learning_parameters['decay_step'] == 0:
            learning_rate *= learning_parameters['decay_rate']

        data = collect_data(pi, gt_dist, training_args["num_decisions"], training_args["fraction_protected"])
        utility, benefit_delta = pi.update(data, learning_rate, training_args["batch_size"])

        yield utility, benefit_delta