import numpy as np

# def iterate_minibatches(X, S, Y, batch_size):
#     indices = np.arange(Y.shape[0])
#     np.random.shuffle(indices)

#     for start_idx in range(0, X.shape[0] - batch_size + 1, batch_size):
#         batch_indices = indices[start_idx:start_idx+batch_size]

#         X_batch = X[batch_indices]
#         S_batch = S[batch_indices]
#         Y_batch = Y[batch_indices]

#         yield X_batch, S_batch, Y_batch

def iterate_minibatches(X, S, Y, batch_size, epochs):
    indices = np.arange(Y.shape[0])

    #for start_idx in range(0, X.shape[0] - batch_size + 1, batch_size):
    for _ in range(0, epochs):
        np.random.shuffle(indices)
        batch_indices = indices[0:batch_size]

        X_batch = X[batch_indices]
        S_batch = S[batch_indices]
        Y_batch = Y[batch_indices]

        yield X_batch, S_batch, Y_batch