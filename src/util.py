import numpy as np

def get_minibatch(features, ground_truth, batch_size):
    indices = np.arange(ground_truth.shape[0])
    np.random.shuffle(indices)
    batch_indices = indices[0:batch_size]

    features_batch = features[batch_indices]
    gt_batch = ground_truth[batch_indices]

    return features_batch, gt_batch