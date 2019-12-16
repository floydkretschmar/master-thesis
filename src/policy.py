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


class BasePolicy():
    def __init__(self, dim_theta, fairness_function, utility_function): 
        self.fairness_function = lambda ips_weight, **fairness_kwargs : ips_weight * fairness_function(fairness_kwargs)
        self.utility_function = lambda ips_weight, **utility_kwargs: ips_weight * utility_function(utility_kwargs)

        self.theta = np.zeros(dim_theta)

    def __call__(self, x, s):
        features = np.concatenate((x, s), axis=1)
        probability = self.calculate_probability(features)
        return np.random.binomial(1, probability)

    def calculate_probability(self, features):
        raise NotImplementedError("Subclass must override calculate probability(features).")

    def calculate_gradient(self, x, s, y, sample_theta, fairness_rate):
        raise NotImplementedError("Subclass must override calculate calculate_gradient(self, x, s, y, sample_theta, fairness_rate).")

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


class LogisticPolicy(BasePolicy):
    def __init__(self, dim_theta, cost_factor, fairness_function): 
        super(LogisticPolicy, self).__init__(
            dim_theta, 
            fairness_function, 
            lambda **utility_kwargs: (self(utility_kwargs["features"]) * (utility_kwargs["y"] - cost_factor)))

        self.feature_map = IdentityFeatureMap(dim_theta)
    
    def calculate_probability(self, features):
        return sigmoid(self.feature_map(features) @ self.theta)

    def calculate_gradient(self, x, s, y, sample_theta, fairness_rate):
        features = np.concatenate((x, s), axis=1)
        phi = self.feature_map(features)
        num_samples = x.shape[0]
        ones = np.ones(num_samples)

        # the inverse propensity scoring weights 1/pi_t-1 = 1 + exp(-(phi_i @ theta_t-1))
        # Shape: (num_samples x 1)
        ips_weight = ones + np.exp(-1 * (phi @ sample_theta))

        # the denominator of the gradient of log pi defined as phi_i/(1+exp(phi_i @ theta_t))
        # Shape: (num_samples x 1)
        denominator = ones + np.exp((phi @ self.theta))

        # calculate the gradient of the utility function
        # Shape: (num_samples x dim_theta)
        grad_utility = (self.utility_function(ips_weight, features=features, y=y) * phi) / denominator

        fairness_params = {
            "x": x, 
            "s": s, 
            "y": y,
            "sample_theta": sample_theta,
            "feature_map": self.feature_map
        }

        # calculate the gradient of the fairness function
        # Shape: (num_samples x dim_theta)
        grad_fairness = (self.fairness_function(ips_weight=ips_weight, **fairness_params) * phi) / denominator
        grad_fairness = fairness_rate * self.fairness_function(ips_weight=ips_weight, **fairness_params) * (grad_fairness)

        # sum the both together, sum over the batch and weigh by the number of samples
        # Shape: (1 x dim_theta)
        gradient = grad_utility + grad_fairness
        gradient = gradient.sum(axis=0) / num_samples

        return gradient