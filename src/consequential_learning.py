import numpy as np

def collect_data(pi, gt_dist, num_samples, fraction_protected):
    x, s = gt_dist.sample_features(num_samples, fraction_protected)
    decisions = pi(x, s)

    pos_decision_idx = np.arange(x.shape[0])
    pos_decision_idx = pos_decision_idx[decisions == 1]

    x = x[pos_decision_idx]
    s = s[pos_decision_idx]

    y = gt_dist.sample_labels(x, s)

    return x, s, y
