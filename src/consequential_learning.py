import numpy as np
from policy import LogisticPolicy
from distribution import GroundTruthDistribution
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

def consequential_learning(fairness_function):    
    """This is the main training loop for consequential learning.

    Keyword arguments:
    fairness_function -- The function defining the fairness penalty.
    """
    gt_dist = GroundTruthDistribution()
    pi = LogisticPolicy(DIM_S + DIM_X, COST_FACTOR, fairness_function)
    for _ in range(0, T):        
        data = collect_data(pi, gt_dist, NUM_DECISIONS)
        pi.update(data, LEARNING_RATE, FAIRNESS_RATE, BATCH_SIZE, NUM_ITERATIONS)

def collect_data(pi, gt_dist, N):
    X = []
    S = []
    Y = []
    for _ in range(0, N):
        x, s = gt_dist.sample()
        features = np.concatenate((x, s), axis=1)
        d = pi(features)
        if d == 1:
            y = gt_dist(x, s)
            X.append(x)
            S.append(s)
            Y.append(y)

    return np.array(X), np.array(S), np.array(Y)