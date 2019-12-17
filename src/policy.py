import sys
import os

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import numpy as np
#pylint: disable=no-name-in-module
from scipy.special import expit as sigmoid
from src.util import get_minibatch


class BasePolicy():
    def __init__(self, dim_theta, fairness_function, utility_function): 
        self.fairness_function = fairness_function
        self.utility_function = utility_function
        self.theta = np.zeros(dim_theta)

    def __call__(self, x, s):
        features = np.concatenate((x, s), axis=1)
        probability = self.calculate_probability(features)
        return np.random.binomial(1, probability)

    def calculate_probability(self, features):
        raise NotImplementedError("Subclass must override calculate probability(features).")

    def calculate_gradient(self, x, s, y, sample_theta):
        raise NotImplementedError("Subclass must override calculate calculate_gradient(self, x, s, y, sample_theta).")

    def calculate_ips_weights_and_log_gradient(self, x, s, y, sample_theta):
        raise NotImplementedError("Subclass must override calculate_ips_weights_and_log_gradient(self, x, s, y, sample_theta).")

    def loss(self, x, s, y):
        raise NotImplementedError("Subclass must override loss(self, x, s, y).")

    def make_decisions(self, X_batch, S_batch):
        decisions = self(X_batch, S_batch)

        # only use data where positive decisions have been made for gradient calculation
        pos_decision_idx = np.arange(X_batch.shape[0])
        pos_decision_idx = pos_decision_idx[decisions == 1]

        return pos_decision_idx

    def update(self, data, learning_rate, batch_size, epochs):
        X, S, Y = data
        sample_theta = self.theta.copy()        

        for _ in range(0, epochs):
            # Get minibatch
            X_batch, S_batch, Y_batch = get_minibatch(X, S, Y, batch_size)

            # make decision according to current policy
            pos_decision_idx = self.make_decisions(X_batch, S_batch)

            # calculate the gradient
            gradient = self.calculate_gradient(X_batch[pos_decision_idx], S_batch[pos_decision_idx], Y_batch[pos_decision_idx], sample_theta)

            # update the parameters
            self.theta += learning_rate * gradient 


class LogisticPolicy(BasePolicy):
    def __init__(self, dim_theta, fairness_rate, cost_factor, fairness_function, feature_map): 
        super(LogisticPolicy, self).__init__(
            dim_theta,
            fairness_function, 
            lambda **utility_kwargs: self(utility_kwargs["x"], utility_kwargs["s"]).reshape(-1, 1) * (utility_kwargs["y"] - cost_factor))

        self.feature_map = feature_map
        self.fairness_rate = fairness_rate
        self.cost_factor = cost_factor
    
    def calculate_probability(self, features):
        return sigmoid(self.feature_map(features) @ self.theta)

    def calculate_gradient(self, x, s, y, sample_theta):
        num_samples = x.shape[0]

        # the inverse propensity scoring weights 1/pi_t-1 = 1 + exp(-(phi_i @ theta_t-1))
        # Shape: (num_samples x 1)
        ips_weight, phi, log_gradient_denominator = self.calculate_ips_weights_and_log_gradient(x, s, sample_theta)

        # the fraction of ips weight and denominator that is used in the utility
        fraction = ips_weight / log_gradient_denominator

        # calculate the gradient of the utility function
        # Shape: (num_samples x dim_theta)
        grad_utility = fraction * (self.utility_function(x=s, s=s, y=y) * phi)

        fairness_params = {
            "x": x, 
            "s": s, 
            "sample_theta": sample_theta,
            "policy": self
        }

        # calculate the gradient of the fairness function
        # Shape: (num_samples x dim_theta)
        if self.fairness_rate > 0:
            grad_fairness = self.fairness_rate * self.fairness_function(**fairness_params, gradient=False) * self.fairness_function(**fairness_params, gradient=True)

        # sum the both together, sum over the batch and weigh by the number of samples
        # Shape: (1 x dim_theta)
        gradient = grad_utility + grad_fairness
        gradient = gradient.sum(axis=0) / num_samples

        return gradient

    def calculate_ips_weights_and_log_gradient(self, x, s, sample_theta):
        """ This function calculates the concrete gradient of the logarithm of pi as well as the concrete
        inverse propensity scroring weights for the logisitic policy.
        
        Args:
            x: The features of the n samples
            s: The sensitive attribute of the n samples
            sample_theta: The parameters of the distribution that was used to sample the n samples.

        Returns:
            ips_weights: the inverse propensity scoring weights 1/pi_t-1 = 1 + exp(-(phi_i @ theta_t-1))
            numerator: numerator of the gradient of log pi which is the same as phi
            denominator: the denominator of the gradient of log pi defined as phi_i/(1+exp(phi_i @ theta_t))
        """
        features = np.concatenate((x, s), axis=1)
        phi = self.feature_map(features)
        ones = np.ones(x.shape)
        sample_theta = sample_theta.reshape(-1, 1)

        return ones + np.exp(-1 * (phi @ sample_theta)), phi, ones + np.exp(phi @ self.theta.reshape(-1, 1))

    def loss(self, x, s, y):
        return 0