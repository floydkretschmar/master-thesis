import sys
import os
import pandas as pd

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import numpy as np
#pylint: disable=no-name-in-module
from scipy.special import expit as sigmoid
from src.util import iterate_minibatches

class BasePolicy():
    """ The base implementation of a policy """

    def __init__(
        self, 
        theta, 
        fairness_gradient_function, 
        benefit_function, 
        utility_function, 
        fairness_rate, 
        use_sensitive_attributes, 
        learn_on_entire_history): 
        """ Initializes a new BasePolicy object.
        
        Args:
            dim_theta: The dimension of the parameters parameterizing the policy model.
            fairness_gradient_function: The callback to the function that defines the fairness penalty of the model.
            benefit_function: The callback that returns the benefit value for a specified set of data.
            utility_function: The callback that returns the utility value for a specified set of data.
            fairness_rate: The fairness rate lambda that regulates the impact of the fairness penalty on the 
            overall utility.
            use_sensitive_attributes: A flag that indicates whether or not the sensitive attributes should be
            used overall or just in evaluating the fairness penalty.
            learn_on_entire_history: A flag that indicates whether or not the policy should be trained on the data
            of all timesteps t < t' or only on the data of the current time step.
        """
        self.fairness_gradient_function = fairness_gradient_function
        self.benefit_function = benefit_function
        self.utility_function = utility_function
        self.use_sensitive_attributes = False
        self.fairness_rate = fairness_rate

        self.use_sensitive_attributes = use_sensitive_attributes
        self.learn_on_entire_history = learn_on_entire_history
        self.data_history = None
        self.theta = theta

    def __call__(self, x, s):
        """ Returns the decisions made by the policy for x and s.
        
        Args:
            x: The features of the n samples
            s: The sensitive attribute of the n samples

        Returns:
            decisions: The decisions.
        """
        features = self._extract_features(x, s)
        probability = self._probability(features)

        return np.random.binomial(1, probability).astype(float)

    def _extract_features(self, x, s):
        """ Extracts the relevant features from the sample.
        
        Args:
            x: The features of the n samples
            s: The sensitive attribute of the n samples

        Returns:
            features: The relevant features of the samples.
        """
        if self.use_sensitive_attributes:
            features = np.concatenate((x, s), axis=1)
        else:
            features = x

        return features

    def _ips_weights(self, x, s, sampling_distribution):
        """ Calculates the inverse propensity scoring weights according to the formula 1/pi_0(e = 1 | x, s).
        
        Args:
            x: The features of the n samples
            s: The sensitive attribute of the n samples
            sampling_distribution: The distribution pi_0 under which the data has been collected.

        Returns:
            ips_weights: The weights for inverse propensity scoring.
        """
        raise NotImplementedError("Subclass must override _ips_weights(self, x, s, sampling_distribution).")

    def _mean_difference(self, target, s):
        """ Calculates the mean difference of the target with regards to the sensitive attribute.
        
        Args:
            target: The target for which the mean difference will be calculated.
            s: The sensitive attribute of the n samples

        Returns:
            mean_difference: The mean difference of the target
        """
        s_idx = np.arange(s.shape[0]).reshape(-1, 1)
        s_0_idx = s_idx[s == 0]
        s_1_idx = s_idx[s == 1]

        if len(s_0_idx) == 0 or len(s_1_idx) == 0:
            return 0.0

        target_s0 = target[s_0_idx].sum(axis=0) / len(s_0_idx)
        target_s1 = target[s_1_idx].sum(axis=0) / len(s_1_idx)

        return target_s0 - target_s1

    def _utility_gradient(self, x, s, y, decisions, ips_weights=None):
        """ Calculates the part of the gradient that is defined by the utility function.
        
        Args:
            x: The features of the n samples
            s: The sensitive attribute of the n samples
            y: The ground truth labels of the n samples
            decisions: The decisions made by the policy based on the features.
            ips_weights: The weights used for inverse propensity scoring. If sampling_data=None 
            no IPS will be applied.

        Returns:
            utility_gradient: The partial gradient of the policy.
        """
        raise NotImplementedError("Subclass must override _utility_gradient(self, x, s, y, decisions, ips_weights=None).")

    def _policy_gradient(self, x, s, y, ips_weights=None):
        """ Calculates the gradient of the policy given the data.
        
        Args:
            x: The features of the n samples
            s: The sensitive attribute of the n samples
            y: The ground truth labels of the n samples
            ips_weights: The weights used for inverse propensity scoring. If sampling_data=None 
            no IPS will be applied.

        Returns:
            gradient: The gradient of the policy.
        """
        # make decision according to current policy
        decisions = self(x, s)

        # calculate utility part of the gradient
        gradient = self._utility_gradient(x, s, y, decisions, ips_weights)

        if self.fairness_rate > 0:
            fairness_gradient = self.fairness_gradient_function(
                x=x, 
                s=s, 
                y=y, 
                ips_weights=ips_weights, 
                decisions=decisions, 
                policy=self)
            grad_fairness = self.fairness_rate * fairness_gradient
            gradient += grad_fairness

        return gradient

    def _probability(self, features):
        """ Calculates the probability of a positiv decision given the specified features.
        
        Args:
            features: The features for which the probability will be calculated

        Returns:
            probability: The Probability of a positive decision.
        """
        raise NotImplementedError("Subclass must override calculate probability(features).")

    def benefit_delta(self, x, s, y, decisions):
        """ Calculates the absolute difference of benefits of the given policy for the provided data.
        
        Args:
            x: The features of the n samples
            s: The sensitive attribute of the n samples
            y: The ground truth labels of the n samples
            decisions: The decisions made by the policy based on the features.

        Returns:
            benefit_delta: The absolute difference in benefits of the policy.
        """    
        return np.absolute(self._mean_difference(self.benefit_function(decisions=decisions, x=x, s=s, y=y), s))

    def copy(self):
        """ Creates a deep copy of the policy.
        
        Returns:
            copy: The created deep copy.
        """
        raise NotImplementedError("Subclass must override copy(self).")

    def update(self, x, s, y, learning_rate, batch_size, epochs):
        """ Updates the policy parameters using stochastic gradient descent.
        
        Args:
            x: The features of the n samples
            s: The sensitive attribute of the n samples
            y: The ground truth labels of the n samples
            learning_rate: The rate with which the parameters will be updated.
            batch_size: The minibatch size of SGD.

        """
        ips_weights = self._ips_weights(x, s, self)
        if self.learn_on_entire_history and self.data_history is None:
            self.data_history = {
                "x": x,
                "s": s,
                "y": y,
                "ips_weights": ips_weights
            }
        elif self.learn_on_entire_history:
            x = np.vstack((self.data_history["x"], x))
            y = np.vstack((self.data_history["y"], y))
            s = np.vstack((self.data_history["s"], s))
            ips_weights = np.vstack((self.data_history["ips_weights"], ips_weights))
            self.data_history["ips_weights"] = ips_weights
            self.data_history["x"] = x
            self.data_history["y"] = y
            self.data_history["s"] = s

        for _ in range(0, epochs):
            # only train if there is a large enough sample size to build at least one full batch
            if x.shape[0] < batch_size:
                break

            # minibatching     
            indices = np.random.permutation(x.shape[0])   
            for batch_start in range(0, len(indices), batch_size):
                batch_end = min(batch_start + batch_size, len(indices))

                X_batch = x[batch_start:batch_end]
                S_batch = s[batch_start:batch_end]
                Y_batch = y[batch_start:batch_end]
                ips_weights_batch = ips_weights[batch_start:batch_end]

                # calculate the gradient
                gradient = self._policy_gradient(X_batch, S_batch, Y_batch, ips_weights_batch)   
                # update the parameters
                self.theta += learning_rate * gradient

    def utility(self, x, s, y, decisions):
        """ Calculates the utility value or the utility gradient according to the utility vaue function callback specified
        in the constructor.
        
        Args:
            x: The features of the n samples
            s: The sensitive attribute of the n samples
            y: The ground truth labels of the n samples
            decisions: The decisions made by the policy based on the features.

        Returns:
            utility: The utility value or gradient of the policy.
        """
        return self.utility_function(decisions=decisions, x=x, s=s, y=y).mean(axis=0)

class LogisticPolicy(BasePolicy):
    """ The implementation of the logistic policy. """

    def __init__(self, theta, fairness_gradient_function, benefit_function, utility_function, feature_map, fairness_rate, use_sensitive_attributes, learn_on_entire_history): 
        super(LogisticPolicy, self).__init__(
            theta,
            fairness_gradient_function,
            benefit_function,
            utility_function,
            fairness_rate,
            use_sensitive_attributes,
            learn_on_entire_history)

        self.feature_map = feature_map    

    def copy(self):
        approx_policy = LogisticPolicy(
            self.theta.shape[0],
            self.fairness_gradient_function, 
            self.benefit_function, 
            self.utility_function, 
            self.feature_map, 
            self.fairness_rate, 
            self.use_sensitive_attributes,
            self.learn_on_entire_history) 
        approx_policy.theta = self.theta.copy()
        return approx_policy

    def _ips_weights(self, x, s, sampling_distribution):
        phi = sampling_distribution.feature_map(sampling_distribution._extract_features(x, s))

        sampling_theta = np.array(sampling_distribution.theta.copy()).reshape(-1,1)
        weights = 1.0 + np.exp(-np.matmul(phi, sampling_theta))
        #weights = 1 / sigmoid(np.matmul(phi, sampling_theta))

        return weights.reshape(-1,1)

    def _utility_gradient(self, x, s, y, decisions, ips_weights=None):
        phi = self.feature_map(self._extract_features(x, s))
        denominator = (1.0 + np.exp(np.matmul(phi, self.theta))).reshape(-1, 1)
        utility = self.utility_function(decisions=decisions, y=y)

        if ips_weights is not None:
            utility = ips_weights * utility       
        
        utility_grad = utility * phi/denominator
        
        return np.mean(utility_grad, axis=0)

    def _probability(self, features):
        return sigmoid(np.matmul(self.feature_map(features), self.theta))
