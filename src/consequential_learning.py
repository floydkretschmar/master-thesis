import numpy as np
from policy import LogisticPolicy
from distribution import GroundTruthDistribution
from util import get_minibatch

'''
pi_0 = init_policy()
for t = 0...T-1 do
    Dt = collect_data(pi_t, N)
    p_t+1 = update_policy(pi_t, Dt, M, B, alpha)
'''

DIM_X = 1
DIM_S = 1
T = 100
NUM_DECISIONS = 500
NUM_ITERATIONS = 200
BATCH_SIZE = 128
LEARNING_RATE = 0.01

def consequential_learning(fairness_function, fairness_gradient_function):
    gt_dist = GroundTruthDistribution()
    pi = LogisticPolicy(DIM_S + DIM_X, fairness_function, fairness_gradient_function)
    for _ in range(0, T):        
        data = collect_data(pi, gt_dist, NUM_DECISIONS)
        pi.update(data, LEARNING_RATE, BATCH_SIZE, NUM_ITERATIONS)

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