import numpy as np

def get_minibatch(X, S, Y, batch_size):
    indices = np.arange(Y.shape[0])
    np.random.shuffle(indices)
    batch_indices = indices[0:batch_size]

    X_batch = X[batch_indices]
    S_batch = S[batch_indices]
    Y_batch = Y[batch_indices]

    return X_batch, S_batch, Y_batch