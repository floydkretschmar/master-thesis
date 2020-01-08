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
        dim_theta, 
        fairness_function, 
        benefit_value_function, 
        utility_value_function, 
        fairness_rate, 
        use_sensitive_attributes, 
        learn_on_entire_history): 
        """ Initializes a new BasePolicy object.
        
        Args:
            dim_theta: The dimension of the parameters parameterizing the policy model.
            fairness_function: The callback to the function that defines the fairness penalty of the model.
            benefit_value_function: The callback that returns the benefit value for a specified set of data.
            utility_value_function: The callback that returns the utility value for a specified set of data.
            fairness_rate: The fairness rate lambda that regulates the impact of the fairness penalty on the 
            overall utility.
            use_sensitive_attributes: A flag that indicates whether or not the sensitive attributes should be
            used overall or just in evaluating the fairness penalty.
            learn_on_entire_history: A flag that indicates whether or not the policy should be trained on the data
            of all timesteps t < t' or only on the data of the current time step.

        Returns:
            decisions: The decisions.
        """
        self.fairness_function = fairness_function
        self.benefit_value_function = benefit_value_function
        self.utility_value_function = utility_value_function
        self.use_sensitive_attributes = False
        self.fairness_rate = fairness_rate

        self.use_sensitive_attributes = use_sensitive_attributes
        self.learn_on_entire_history = learn_on_entire_history
        self.data_history = None
        self.theta = np.zeros(dim_theta)

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

    def _benefit_difference(self, x, s, y, decisions, gradient=False, ips_weights=None):
        """ Calculates the difference of benefits of the given policy for the provided data.
        
        Args:
            x: The features of the n samples
            s: The sensitive attribute of the n samples
            y: The ground truth labels of the n samples
            decisions: The decisions made by the policy based on the features.
            ips_weights: The weights used for inverse propensity scoring. If sampling_data=None 
            no IPS will be applied.

        Returns:
            benefit_difference: The difference in benefits of the policy.
        """
        s_idx = np.arange(s.shape[0]).reshape(-1, 1)
        s_0_idx = s_idx[s == 0]
        s_1_idx = s_idx[s == 1]

        if ips_weights is not None:
            ips_weights_s0 = ips_weights[s_0_idx]
            ips_weights_s1 = ips_weights[s_1_idx]
        else:
            ips_weights_s0 = None
            ips_weights_s1 = None

        benefit_s0 = self._calculate_expectation(x[s_0_idx], s[s_0_idx], self.benefit_value_function(decisions_s=decisions[s_0_idx], y_s=y[s_0_idx]), gradient, ips_weights_s0)
        benefit_s1 = self._calculate_expectation(x[s_1_idx], s[s_1_idx], self.benefit_value_function(decisions_s=decisions[s_1_idx], y_s=y[s_1_idx]), gradient, ips_weights_s1)

        return benefit_s0 - benefit_s1

    def _calculate_expectation(self, x, s, function_value, gradient=False, ips_weights=None):
        """ Calculates the expectation over the underlying probability distribition of the policy for the 
        specified function value E[target] or the gradient of the policy E[target * \log \grad policy(e | x, s)].
        If specified IPS is applied according to the formula target/pi_theta(e = 1 | x, s) to the function
        value.
        
        Args:
            x: The features of the n samples
            s: The sensitive attribute of the n samples
            function_value: The function value for which the expectation will be calculated.
            gradient: The flag specifying whether the gradient should be calculated instead of the function 
            value expectation.
            ips_weights: The weights used for inverse propensity scoring. If sampling_data=None 
            no IPS will be applied.

        Returns:
            expectation or gradient: The expectation over the policy with regard to the specified function 
            value or its gradient.
        """
        raise NotImplementedError("Subclass must override _calculate_expectation(self, x, s, target, gradient=False, sampling_theta=None).")

    def _calculate_ips_weights(self, x, s, sampling_distribution):
        """ Calculates the inverse propensity scoring weights according to the formula 1/pi_theta(e = 1 | x, s).
        
        Args:
            x: The features of the n samples
            s: The sensitive attribute of the n samples
            sampling_distribution: The distribution under which the data has been collected.

        Returns:
            ips_weights: The weights for inverse propensity scoring.
        """
        raise NotImplementedError("Subclass must override _calculate_ips_weights(self, x, s, sampling_distribution).")

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

    def _fairness_function(self, x, s, y, decisions, gradient=False, ips_weights=None):
        """ Calls the external fairness function callback.
        
        Args:
            x: The features of the n samples
            s: The sensitive attribute of the n samples
            y: The ground truth labels of the n samples
            decisions: The decisions made by the policy based on the features.
            gradient: The flag specifying whether the gradient should be calculated instead of the function 
            value expectation.
            ips_weights: The weights used for inverse propensity scoring. If sampling_data=None 
            no IPS will be applied.

        Returns:
            ips_weights: The weights for inverse propensity scoring.
        """
        return self.fairness_function(
            x=x, 
            s=s, 
            y=y, 
            ips_weights=ips_weights, 
            decisions=decisions, 
            gradient=gradient, 
            policy=self)

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
        gradient = self._calculate_expectation(x, s, self.utility_value_function(decisions=decisions, y=y), gradient=True, ips_weights=ips_weights) 

        if self.fairness_rate > 0:
            fairness_value = self._fairness_function(x, s, y, decisions, gradient=False, ips_weights=ips_weights)
            fairness_gradient_value = self._fairness_function(x, s, y, decisions, gradient=True, ips_weights=ips_weights)
            grad_fairness = self.fairness_rate * fairness_value * fairness_gradient_value
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
        return np.absolute(self._benefit_difference(x, s, y, decisions))

    def copy(self):
        """ Creates a deep copy of the policy.
        
        Returns:
            copy: The created deep copy.
        """
        raise NotImplementedError("Subclass must override copy(self).")

    def regularized_utility(self, x, s, y, decisions):
        """ Calculates the overall utility of the policy regularized by the fairness constraint.
        
        Args:
            x: The features of the n samples
            s: The sensitive attribute of the n samples
            y: The ground truth labels of the n samples
            decisions: The decisions made by the policy based on the features.

        Returns:
            utility: The utility of the policy.
        """
        regularized_utility = self._calculate_expectation(x, s, self.utility_value_function(decisions=decisions, y=y))

        if self.fairness_rate > 0:
            fairness_value = self._fairness_function(x, s, y, decisions)
            fairness_penalty = (self.fairness_rate/2) * (fairness_value**2)
            regularized_utility -= fairness_penalty

        return regularized_utility

    def update(self, x, s, y, learning_rate, batch_size, epochs):
        """ Updates the policy parameters using stochastic gradient descent.
        
        Args:
            x: The features of the n samples
            s: The sensitive attribute of the n samples
            y: The ground truth labels of the n samples
            learning_rate: The rate with which the parameters will be updated.
            batch_size: The minibatch size of SGD.

        """
        ips_weights = self._calculate_ips_weights(x, s, self)
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

        indices = np.arange(x.shape[0])

        for _ in range(0, epochs):
            # minibatching
            np.random.shuffle(indices)            
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
        return self._calculate_expectation(x, s, self.utility_value_function(decisions=decisions, y=y))

class LogisticPolicy(BasePolicy):
    """ The implementation of the logistic policy. """

    def __init__(self, dim_theta, fairness_function, benefit_value_function, utility_value_function, feature_map, fairness_rate, use_sensitive_attributes, learn_on_entire_history): 
        super(LogisticPolicy, self).__init__(
            dim_theta,
            fairness_function,
            benefit_value_function,
            utility_value_function,
            fairness_rate,
            use_sensitive_attributes,
            learn_on_entire_history)

        self.feature_map = feature_map    

    def copy(self):
        approx_policy = LogisticPolicy(
            self.theta.shape[0],
            self.fairness_function, 
            self.benefit_value_function, 
            self.utility_value_function, 
            self.feature_map, 
            self.fairness_rate, 
            self.use_sensitive_attributes,
            self.learn_on_entire_history) 
        approx_policy.theta = self.theta.copy()
        return approx_policy

    def _probability(self, features):
        return sigmoid(np.matmul(self.feature_map(features), self.theta))

    def _calculate_expectation(self, x, s, function_value, gradient=False, ips_weights=None):
        if gradient:
            phi = self.feature_map(self._extract_features(x, s))
            ones = np.ones((function_value.shape[0], 1))
            distance = np.matmul(phi, self.theta.reshape(-1, 1))

            # when ips weighting: divide ips weights by gradient weights first to stabelize
            # big values
            if ips_weights is not None:
                #weights = ips_weights / (ones + np.exp(distance))
                weights = ips_weights * sigmoid(-distance)
                function_value = function_value * weights
            else:
                function_value = function_value / (ones + np.exp(distance))

            function_value = phi * function_value
        elif ips_weights is not None:
            function_value *= ips_weights

        return function_value.sum(axis=0) / function_value.shape[0]

    def _calculate_ips_weights(self, x, s, sampling_distribution):
        phi = sampling_distribution.feature_map(sampling_distribution._extract_features(x, s))
        ones = np.ones((x.shape[0], 1))

        sampling_theta = sampling_distribution.theta.copy().reshape(-1, 1)
        exp = np.exp(-np.matmul(phi, sampling_theta))
        weights = (ones + exp)

        #weights = 1 / sigmoid(np.matmul(phi, sampling_theta))
        return weights