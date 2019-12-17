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

def train(DIM_S, DIM_X, DIM_THETA, COST_FACTOR, NUM_DECISIONS, FRACTION_PROTECTED, LEARNING_RATE, FAIRNESS_RATE, BATCH_SIZE, NUM_ITERATIONS, T, fairness_function, feature_map):
    gt_dist = SplitDistribution()
    pi = LogisticPolicy(DIM_THETA, FAIRNESS_RATE, COST_FACTOR, fairness_function, feature_map)
    for i in range(0, T):        
        data = collect_data(pi, gt_dist, NUM_DECISIONS, FRACTION_PROTECTED)
        pi.update(data, LEARNING_RATE, BATCH_SIZE, NUM_ITERATIONS)
        print("Iteration {}".format(i))