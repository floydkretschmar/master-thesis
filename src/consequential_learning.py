import numpy as np
from policy import LogisticPolicy
from distribution import GroundTruthDistribution

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

if __name__== "__main__":
    gt_dist = GroundTruthDistribution()
    pi = LogisticPolicy(DIM_S + DIM_X, NUM_ITERATIONS, BATCH_SIZE)
    for t in range(0, T):
        data = collect_data(pi, NUM_DECISIONS)
        pi.update(data, LEARNING_RATE)

def collect_data(pi, gt_dist, N):
    X = []
    S = []
    Y = []
    for i in range(0, N):
        x, s = gt_dist.sample()
        d = pi(x, s)
        if d == 1:
            y = gt_dist(x, s)
            X.append(x)
            S.append(s)
            Y.append(y)

    return np.array(X), np.array(S), np.array(Y)