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

    @property
    def parameters(self):
        """ Returns the model parameters needed to restore the model to its current state.

        Returns:
            parameters: A dictionary of model parameters.
        """
        raise NotImplementedError("Subclass must override parameters(self).")

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
        raise NotImplementedError("Subclass must override _probability(features).")

    def copy(self):
        """ Creates a deep copy of the policy.
        
        Returns:
            copy: The created deep copy.
        """
        raise NotImplementedError("Subclass must override copy(self).")

    @staticmethod
    def optimizer(policy, optimization_target):
        """ Returns the fitting optimizer that optimizes the specified policy.

        Args:
            policy: The policy that will be optimized by the optimizer.
            optimization_target: The optimization target of the optimizer.

        Returns:
            optimizer: The optimizer that optimizes the specified policy according to the specified optimization target.
        """
        raise NotImplementedError(
            "Subclass must override optimizer(policy, optimization_target).")


class ManualGradientPolicy(BasePolicy):
    def __init__(self, use_sensitive_attributes, theta):
        super().__init__(use_sensitive_attributes)
        self._theta = np.array(theta)

    @property
    def theta(self):
        """ Returns the paramters of the policy. """
        return self._theta

    @theta.setter
    def theta(self, value):
        """ Sets the paramters of the policy. """
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

    @staticmethod
    def optimizer(policy, optimization_target):
        """ Returns the optimizer that optimizes a ManualGradientPolicy.

        Args:
            policy: The ManualGradientPolicy that will be optimized by the optimizer.
            optimization_target: The optimization target of type DifferentiableOptimizationTarget that is used to
            optimize the policy.

        Returns:
            optimizer: The ManualStochasticGradientOptimizer that optimizes the specified policy according to the
            specified optimization target.
        """
        return ManualStochasticGradientOptimizer(policy, optimization_target)


class LogisticPolicy(ManualGradientPolicy):
    """ The implementation of the logistic policy. """

    def __init__(self, theta, feature_map, use_sensitive_attributes):
        super(LogisticPolicy, self).__init__(use_sensitive_attributes, theta)
        self.feature_map = feature_map

    @property
    def parameters(self):
        return self.theta.tolist()

    def _probability(self, features):
        return sigmoid(np.matmul(self.feature_map(features), self.theta))

    def copy(self):
        approx_policy = LogisticPolicy(
            deepcopy(self.theta),
            deepcopy(self.feature_map),
            self.use_sensitive_attributes)
        return approx_policy

    def log_policy_gradient(self, x, s):
        phi = self.feature_map(self._extract_features(x, s))
        return phi / np.expand_dims(1.0 + np.exp(np.matmul(phi, self.theta)), axis=1)
