import numpy as np
from policy import LogisticPolicy
from distribution import SplitDistribution
from util import get_minibatch

DIM_X = 1
DIM_S = 1
T = 100
NUM_DECISIONS = 500
NUM_ITERATIONS = 200
BATCH_SIZE = 128
LEARNING_RATE = 0.01
FAIRNESS_RATE = 0.5
COST_FACTOR = 0.6
FRACTION_PROTECTED = 0.3

def consequential_learning(fairness_function):    
    """This is the main training loop for consequential learning.

    Keyword arguments:
    fairness_function -- The function defining the fairness penalty.
    """
    gt_dist = SplitDistribution()
    pi = LogisticPolicy(DIM_S + DIM_X, COST_FACTOR, fairness_function)
    for _ in range(0, T):        
        data = collect_data(pi, gt_dist, NUM_DECISIONS, FRACTION_PROTECTED)
        pi.update(data, LEARNING_RATE, FAIRNESS_RATE, BATCH_SIZE, NUM_ITERATIONS)

def collect_data(pi, gt_dist, num_samples, fraction_protected):
    x, s = gt_dist.sample_features(num_samples, fraction_protected)
    decisions = pi(x, s)

    pos_decision_idx = np.arange(x.shape[0])
    pos_decision_idx = pos_decision_idx[decisions == 1]

    x = x[pos_decision_idx]
    s = s[pos_decision_idx]

    y = gt_dist.sample_labels(x, s)

    return x, s, y