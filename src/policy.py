import numpy as np
from scipy.special import expit as sigmoid
from util import get_minibatch

class BaseFeatureMap():
    """ The feature map phi: R^d x {0,1} -> R^m that maps the feature vector and sensitive attribute of an
    individual into the feature space of the parameters theta"""

    def __init__(self, dim_theta):
        self.dim_theta = dim_theta

    def __call__(self, features):
        return self.map(features)

    def map(self, features):
        raise NotImplementedError("Subclass must override map(x).")

class IdentityFeatureMap(BaseFeatureMap):
    """ The feature map phi as an identity mapping"""

    def __init__(self, dim_theta):
        super(IdentityFeatureMap, self).__init__(dim_theta)

    def map(self, features):
        return features

class LogisticPolicy():
    def __init__(self, dim_theta, fairness_function, fairness_gradient_function): 
        self.fairness_function = fairness_function
        self.fairness_gradient_function = fairness_gradient_function

        self.theta = np.zeros(dim_theta)
        self.feature_map = IdentityFeatureMap(dim_theta)
    
    def __call__(self, features):
        probability = sigmoid(self.feature_map(features) @ self.theta)
        return np.random.binomial(1, probability)

    def update(self, data, learning_rate, batch_size, epochs):
        X, S, Y = data
        features = np.concatenate((X, S), axis=1)
        sample_theta = self.theta.clone()        

        for _ in range(0, epochs):
            # Get minibatch
            batch_features, batch_gt = get_minibatch(features, Y, batch_size)

            # make decision according to current policy
            decisions = self(batch_features)

            # only use data where positive decisions have been made for gradient calculation
            pos_decision_idx = np.arange(batch_gt.shape[0])
            pos_decision_idx = pos_decision_idx[decisions == 1]

            # calculate the gradient
            gradient = self.calculate_gradient(batch_features[pos_decision_idx], batch_gt[pos_decision_idx], sample_theta)

            # update the parameters
            self.theta += learning_rate * gradient / pos_decision_idx.shape[0]

    def calculate_gradient(self, features, gt, sample_theta):
        return 0