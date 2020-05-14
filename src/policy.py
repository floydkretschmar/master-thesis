import os
import sys

root_path = os.path.abspath(os.path.join('..'))
if root_path not in sys.path:
    sys.path.append(root_path)

import numpy as np
from copy import deepcopy
# pylint: disable=no-name-in-module
from src.util import sigmoid, get_random
from src.optimization import ManualStochasticGradientOptimizer


# TODO: Add Pytorch supporting policy (e.g. simple MLP)

class BasePolicy:
    """ The base implementation of a policy """

    def __init__(self, use_sensitive_attributes):
        """ Initializes a new BasePolicy object.
        
        Args:
            fairness_function: The callback to the function that defines the fairness penalty of the model.
            utility_function: The callback that returns the utility value for a specified set of data.
            fairness_rate: The fairness rate lambda that regulates the impact of the fairness penalty on the 
            overall utility.
            use_sensitive_attributes: A flag that indicates whether or not the sensitive attributes should be
            used overall or just in evaluating the fairness penalty.
        """
        self.use_sensitive_attributes = use_sensitive_attributes

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

        return np.expand_dims(get_random().binomial(1, probability).astype(float), axis=1), np.expand_dims(probability,
                                                                                                           axis=1)

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

    def _probability(self, features):
        """ Calculates the probability of a positiv decision given the specified features.
        
        Args:
            features: The features for which the probability will be calculated

        Returns:
            probability: The Probability of a positive decision.
        """
        raise NotImplementedError("Subclass must override calculate probability(features).")

    def copy(self):
        """ Creates a deep copy of the policy.
        
        Returns:
            copy: The created deep copy.
        """
        raise NotImplementedError("Subclass must override copy(self).")

    def get_model_parameters(self):
        """ Returns the model parameters needed to restore the model to its current state.
        
        Returns:
            parameters: A dictionary of model parameters.
        """
        raise NotImplementedError("Subclass must override get_model_parameters(self).")

    def ips_weights(self, x, s):
        """ Calculates the inverse propensity scoring weights according to the formula 1/pi_0(e = 1 | x, s).

        Args:
            x: The features of the n samples
            s: The sensitive attribute of the n samples

        Returns:
            ips_weights: The weights for inverse propensity scoring.
        """
        raise NotImplementedError("Subclass must override _ips_weights(self, x, s, sampling_distribution).")

    def optimizer(self, optimization_target):
        raise NotImplementedError(
            "Subclass must override optimizer(self, optimization_target).")

    def reset(self):
        raise NotImplementedError("Subclass must override reset(self).")


class ManualGradientPolicy(BasePolicy):
    def __init__(self, use_sensitive_attributes, theta):
        super().__init__(use_sensitive_attributes)
        self._theta = self.initialize_theta(theta)
        self._initial_theta = deepcopy(self._theta)

    def initialize_theta(self, theta):
        return np.array(theta)

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, value):
        assert self.theta.shape == value.shape
        self._theta = value

    def log_policy_gradient(self, x, s):
        """ Calculates the gradient of the logarithm of the current policy used to calculate the utility
        and fairness gradients.
        
        Args:
            x: The features of the n samples
            s: The sensitive attribute of the n samples
            sampling_distribution: The distribution pi_0 under which the data has been collected.

        Returns:
            log_gradient: The gradient of the logarithm of the current policy.
        """
        raise NotImplementedError("Subclass must override log_policy_gradient(self, x, s).")

    def optimizer(self, optimization_target):
        return ManualStochasticGradientOptimizer(self, optimization_target)

    def reset(self):
        self._theta = deepcopy(self._initial_theta)


class LogisticPolicy(ManualGradientPolicy):
    """ The implementation of the logistic policy. """

    def __init__(self, theta, feature_map, use_sensitive_attributes):
        super(LogisticPolicy, self).__init__(use_sensitive_attributes, theta)
        self.feature_map = feature_map

    def copy(self):
        approx_policy = LogisticPolicy(
            self.theta.copy(),
            self.feature_map,
            self.use_sensitive_attributes)
        return approx_policy

    def _probability(self, features):
        return sigmoid(np.matmul(self.feature_map(features), self.theta))

    def get_model_parameters(self):
        return self.theta.tolist()

    def ips_weights(self, x, s):
        phi = self.feature_map(self._extract_features(x, s))

        sampling_theta = np.expand_dims(self.theta, axis=1)
        weights = 1.0 + np.exp(-np.matmul(phi, sampling_theta))

        return weights

    def log_policy_gradient(self, x, s):
        phi = self.feature_map(self._extract_features(x, s))
        return phi / np.expand_dims(1.0 + np.exp(np.matmul(phi, self.theta)), axis=1)
