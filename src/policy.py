import numpy as np
#pylint: disable=no-name-in-module
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
    def __init__(self, dim_theta, cost_factor, fairness_function, fairness_gradient_function): 
        self.fairness_function = fairness_function
        self.fairness_gradient_function = fairness_gradient_function
        self.cost_factor = cost_factor

        self.theta = np.zeros(dim_theta)
        self.feature_map = IdentityFeatureMap(dim_theta)
    
    def __call__(self, x, s):
        features = np.concatenate((x, s), axis=1)
        probability = sigmoid(self.feature_map(features) @ self.theta)
        return np.random.binomial(1, probability)

    def update(self, data, learning_rate, fairness_rate, batch_size, epochs):
        X, S, Y = data
        sample_theta = self.theta.clone()        

        for _ in range(0, epochs):
            # Get minibatch
            X_batch, S_batch, Y_batch = get_minibatch(X, S, Y, batch_size)

            # make decision according to current policy
            decisions = self(X_batch, S_batch)

            # only use data where positive decisions have been made for gradient calculation
            pos_decision_idx = np.arange(Y_batch.shape[0])
            pos_decision_idx = pos_decision_idx[decisions == 1]

            # calculate the gradient
            gradient = self.calculate_gradient(X_batch[pos_decision_idx], S_batch[pos_decision_idx], Y_batch[pos_decision_idx], sample_theta, fairness_rate)

            # update the parameters
            self.theta += learning_rate * gradient 

    def calculate_gradient(self, x, s, y, sample_theta, fairness_rate):
        features = np.concatenate((x, s), axis=1)
        phi = self.feature_map(features)
        num_samples = x.shape[0]
        ones = np.ones(num_samples)

        numerator = ones + np.exp(-1 * (phi @ sample_theta))
        denominator = ones + np.exp((phi @ self.theta))

        difference = numerator / denominator

        # calculate the gradient of the utility function (always the same)
        grad_utility = difference * self(features) * (y - self.cost_factor) * phi

        # calculate the gradient of the fairness function (changable)
        grad_fairness = difference * self.fairness_gradient_function(x=x, s=s, y=y, sample_theta=sample_theta) * phi
        grad_fairness = fairness_rate * self.fairness_function(x=x, s=s, y=y) * (grad_fairness)

        # sum the both together, sum over the batch and weigh by the number of samples
        gradient = grad_utility + grad_fairness
        gradient = gradient.sum(axis=0) / num_samples

        return gradient