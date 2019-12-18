import sys
import os

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import numpy as np
#pylint: disable=no-name-in-module
from scipy.special import expit as sigmoid
from src.util import iterate_minibatches


class BasePolicy():
    def __init__(self, fairness_function, dim_x, dim_s=None): 
        self.fairness_function = fairness_function
        self.use_s = False

        dim_theta = dim_x
        if dim_s is not None:
            self.use_s = True
            dim_theta += dim_s

        self.theta = np.zeros(dim_theta)
    
    def benefit_function(self, x_s, s, sample_theta, gradient):
        raise NotImplementedError("Subclass must override calculate benefit_function(self, x_s, s, sample_theta, gradient).")

    def __call__(self, features):
        probability = self.calculate_probability(features)
        return np.random.binomial(1, probability)

    def calculate_benefit_delta(self, x, s, sample_theta):
        return np.absolute(self.calculate_benefit_difference(x, s, sample_theta, False, self.use_s)).mean()

    def calculate_benefit_difference(self, x, s, sample_theta, gradient, use_s):
        raise NotImplementedError("Subclass must override calculate benefit_difference(self, s, x, sample_theta, gradient).")

    def calculate_probability(self, features):
        raise NotImplementedError("Subclass must override calculate probability(features).")

    def calculate_gradient(self, x, s, y, sample_theta):
        raise NotImplementedError("Subclass must override calculate calculate_gradient(self, x, s, y, sample_theta).")

    def calculate_ips_weights_and_log_gradient(self, x, s, y, sample_theta):
        raise NotImplementedError("Subclass must override calculate_ips_weights_and_log_gradient(self, x, s, y, sample_theta).")

    def calculate_utility(self, x, s, y, sample_theta):
        raise NotImplementedError("Subclass must override loss(self, x, s, y, sample_theta).")

    def make_decisions(self, X_batch, S_batch, use_s):
        if self.use_s:
            features = np.concatenate((X_batch, S_batch), axis=1)
        else:
            features = X_batch

        decisions = self(features)

        return decisions

    def update(self, data, learning_rate, batch_size):
        x, s, y = data
        sample_theta = self.theta.copy()        

        for X_batch, S_batch, Y_batch in iterate_minibatches(x, s, y, batch_size):
            # make decision according to current policy
            decisions = self.make_decisions(X_batch, S_batch, self.use_s)

            # only use data where positive decisions have been made for gradient calculation
            pos_decision_idx = np.arange(X_batch.shape[0])
            pos_decision_idx = pos_decision_idx[decisions == 1]

            # calculate the gradient
            gradient = self.calculate_gradient(X_batch[pos_decision_idx], S_batch[pos_decision_idx], Y_batch[pos_decision_idx], sample_theta)

            # update the parameters
            self.theta += learning_rate * gradient 

        return self.calculate_utility(x, s, y, sample_theta), self.calculate_benefit_delta(x, s, sample_theta)

    def utility_function(self, x, s, y, sample_theta, gradient):
        raise NotImplementedError("Subclass must override calculate utility_function(self, x, s, y, sample_theta, gradient).")


class LogisticPolicy(BasePolicy):
    def __init__(self, fairness_rate, cost_factor, fairness_function, feature_map, dim_x, dim_s=None): 
        super(LogisticPolicy, self).__init__(
            fairness_function,
            dim_x,
            dim_s)

        self.feature_map = feature_map
        self.fairness_rate = fairness_rate
        self.cost_factor = cost_factor

    def benefit_function(self, x_s, s, sample_theta, gradient, use_s):
        ips_weight, phi, log_gradient_denominator = self.calculate_ips_weights_and_log_gradient(x_s, s, sample_theta)

        decisions = self.make_decisions(x_s, s, use_s).reshape(-1, 1)

        if gradient:
            grad_benefit = ((ips_weight/log_gradient_denominator) * decisions * phi).sum(axis=0) / x_s.shape[0]
            return grad_benefit
        else:
            benefit = (ips_weight * decisions).sum(axis=0) / x_s.shape[0]
            return benefit

    def calculate_benefit_difference(self, x, s, sample_theta, gradient, use_s):
        pos_decision_idx = np.arange(s.shape[0]).reshape(-1, 1)

        s_0_idx = pos_decision_idx[s == 0]
        s_1_idx = pos_decision_idx[s == 1]

        return self.benefit_function(x[s_0_idx], s[s_0_idx], sample_theta, gradient, use_s) - self.benefit_function(x[s_1_idx], s[s_1_idx], sample_theta, gradient, use_s)
    
    def calculate_probability(self, features):
        return sigmoid(self.feature_map(features) @ self.theta)

    def calculate_gradient(self, x, s, y, sample_theta):
        # get the gradient value of the utility function
        gradient = self.utility_function(x, s, y, sample_theta, True)

        fairness_params = {
            "x": x, 
            "s": s, 
            "sample_theta": sample_theta,
            "policy": self,
            "use_s": self.use_s
        }

        # get the gradient value of the fairness function
        # Shape: (num_samples x dim_theta)
        if self.fairness_rate > 0:
            grad_fairness = self.fairness_rate * self.fairness_function(**fairness_params, gradient=False) * self.fairness_function(**fairness_params, gradient=True)
            gradient -= grad_fairness

        gradient = gradient.mean(axis=0)
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
        if self.use_s:
            features = np.concatenate((x, s), axis=1)
        else:
            features = x

        phi = self.feature_map(features)
        ones = np.ones(x.shape)
        sample_theta = sample_theta.reshape(-1, 1)

        return ones + np.exp(-1 * (phi @ sample_theta)), phi, ones + np.exp(phi @ self.theta.reshape(-1, 1))

    def calculate_utility(self, x, s, y, sample_theta):
        fairness_params = {
            "x": x, 
            "s": s, 
            "sample_theta": sample_theta,
            "policy": self,
            "gradient": False,
            "use_s": self.use_s
        }
        fairness_pen = (self.fairness_rate * self.fairness_function(**fairness_params)**2)/2
        #print("Fairness penalty: {}".format(fairness_pen))
        return (self.utility_function(x=s, s=s, y=y, sample_theta=sample_theta, gradient=False) - (fairness_pen)).mean()


    def utility_function(self, x, s, y, sample_theta, gradient):
        ips_weight, phi, log_gradient_denominator = self.calculate_ips_weights_and_log_gradient(x, s, sample_theta)

        decisions = self.make_decisions(x, s, self.use_s).reshape(-1, 1)
        utility_value = decisions * (y - self.cost_factor)

        if gradient:
            # calculate the gradient of the utility function
            # Shape: (num_samples x dim_theta)
            return (ips_weight * utility_value * phi) / log_gradient_denominator
        else:
            # calculate the the utility function value
            # Shape: (num_samples x 1)
            return ips_weight * utility_value

        