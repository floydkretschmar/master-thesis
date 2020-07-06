import os
import sys

root_path = os.path.abspath(os.path.join('..'))
if root_path not in sys.path:
    sys.path.append(root_path)

import numpy as np
import abc
import torch
import torch.nn as nn

from copy import deepcopy
# pylint: disable=no-name-in-module
from src.util import sigmoid, to_device, from_device
from src.optimization import ManualStochasticGradientOptimizer, PytorchStochasticGradientOptimizer
from src.functions import ADAM, SGD


# TODO: Add Pytorch supporting policy (e.g. simple MLP)

class BasePolicy(abc.ABC):
    """ The base implementation of a policy """
    def __init__(self, use_sensitive_attributes):
        """ Initializes a new BasePolicy object.

        Args:
            use_sensitive_attributes: A flag that indicates whether or not the sensitive attributes should be
            used overall or just in evaluating the fairness penalty.
        """
        self._use_sensitive_attributes = use_sensitive_attributes

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
        decisions = self._decision(probability)

        return decisions, probability

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
        if self._use_sensitive_attributes:
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

    def _decision(self, probability):
        raise NotImplementedError("Subclass must override _decision(features).")

    def copy(self):
        """ Creates a deep copy of the policy.
        
        Returns:
            copy: The created deep copy.
        """
        raise NotImplementedError("Subclass must override copy(self).")

    def optimizer(self, optimization_target, **optimizer_args):
        """ Returns the fitting optimizer that optimizes the specified policy.

        Args:
            policy: The policy that will be optimized by the optimizer.
            optimization_target: The optimization target of the optimizer.

        Returns:
            optimizer: The optimizer that optimizes the specified policy according to the specified optimization target.
        """
        raise NotImplementedError(
            "Subclass must override optimizer(policy, optimization_target).")


class ManualGradientPolicy(BasePolicy, abc.ABC):
    def __init__(self, use_sensitive_attributes, theta):
        super().__init__(use_sensitive_attributes)
        self._theta = np.array(theta)

    @property
    def parameters(self):
        return deepcopy(self._theta)

    @parameters.setter
    def parameters(self, value):
        """ Sets the paramters of the policy. """
        assert self._theta.shape == value.shape
        self._theta = deepcopy(value)

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

    def optimizer(self, optimization_target, **optimizer_args):
        """ Returns the optimizer that optimizes a ManualGradientPolicy.

        Args:
            policy: The ManualGradientPolicy that will be optimized by the optimizer.
            optimization_target: The optimization target of type DifferentiableOptimizationTarget that is used to
            optimize the policy.

        Returns:
            optimizer: The ManualStochasticGradientOptimizer that optimizes the specified policy according to the
            specified optimization target.
        """
        fairness_optimizer_function = optimizer_args["fairness_training_algorithm"] \
            if "fairness_training_algorithm" in optimizer_args and optimizer_args["fairness_training_algorithm"] is not None \
            else ADAM
        policy_optimizer_function = optimizer_args["policy_training_algorithm"] \
            if "policy_training_algorithm" in optimizer_args and optimizer_args["policy_training_algorithm"] is not None \
            else ADAM
        return ManualStochasticGradientOptimizer(self,
                                                 optimization_target,
                                                 fairness_optimizer_function,
                                                 policy_optimizer_function)


class LogisticPolicy(ManualGradientPolicy):
    """ The implementation of the logistic policy. """

    def __init__(self, feature_map, use_sensitive_attributes):
        super(LogisticPolicy, self).__init__(use_sensitive_attributes, np.zeros(feature_map.dim_theta))
        self.feature_map = feature_map

    def _probability(self, features):
        return sigmoid(np.matmul(self.feature_map(features), self._theta)).reshape(-1, 1)

    def _decision(self, probability):
        decisions = np.random.binomial(1, probability).astype(float)
        return decisions.reshape(-1, 1)

    def copy(self):
        approx_policy = LogisticPolicy(
            deepcopy(self.feature_map),
            self._use_sensitive_attributes)
        return approx_policy

    def log_policy_gradient(self, x, s):
        phi = self.feature_map(self._extract_features(x, s))
        return phi / np.expand_dims(1.0 + np.exp(np.matmul(phi, self._theta)), axis=1)


class PytorchPolicy(BasePolicy, abc.ABC):
    def __init__(self, use_sensitive_attributes, bias=True, **network_arguments):
        super().__init__(use_sensitive_attributes)
        self.bias = bias
        self.network = self._create_network(bias=bias, **network_arguments)

    def _create_network(self, **network_parameters):
        raise NotImplementedError("Subclass must override _create_network(self, **network_parameters).")

    def _probability(self, features):
        features = to_device(features)
        probability = self.network(features)
        probability = from_device(probability)
        return probability.reshape(-1, 1)

    def _decision(self, probability):
        return torch.bernoulli(probability).reshape(-1, 1)

    @property
    def parameters(self):
        return self.network.parameters(), self.network.state_dict()

    @parameters.setter
    def parameters(self, value):
        self.network.load_state_dict(value)

    def optimizer(self, optimization_target, **optimizer_args):
        fairness_optimizer_function = optimizer_args["fairness_training_algorithm"] \
            if "fairness_training_algorithm" in optimizer_args and optimizer_args["fairness_training_algorithm"] is not None \
            else ADAM
        pytorch_optimizer_constructor = optimizer_args["policy_training_algorithm"] \
            if "policy_training_algorithm" in optimizer_args and optimizer_args["policy_training_algorithm"] is not None \
            else torch.optim.Adam
        return PytorchStochasticGradientOptimizer(self,
                                                  optimization_target,
                                                  fairness_optimizer_function,
                                                  pytorch_optimizer_constructor)


class NeuralNetworkPolicy(PytorchPolicy):
    """ The implementation of the neural network policy. """
    class Network(nn.Module):
        def __init__(self, input_size, bias=True):
            super(NeuralNetworkPolicy.Network, self).__init__()
            self.network = nn.Sequential(nn.Linear(input_size, 128, bias=bias),
                                         nn.ReLU(),
                                         nn.Linear(128, 128, bias=bias),
                                         nn.ReLU(),
                                         nn.Linear(128, 128, bias=bias),
                                         nn.ReLU(),
                                         nn.Linear(128, 1, bias=bias),
                                         nn.Sigmoid())

        def forward(self, features):
            return self.network(features)

        def get_name(self):
            return "NeuralNetworkPolicy"

    def __init__(self, input_size, use_sensitive_attributes, bias=True):
        super(NeuralNetworkPolicy, self).__init__(use_sensitive_attributes=use_sensitive_attributes,
                                                  bias=bias,
                                                  input_size=input_size)
        self.input_size = input_size

    def _create_network(self, **network_parameters):
        network = NeuralNetworkPolicy.Network(network_parameters["input_size"], network_parameters["bias"])
        return to_device(network)

    def copy(self):
        copy = NeuralNetworkPolicy(self.input_size, self.bias)
        copy.network = deepcopy(self.network)
        return copy