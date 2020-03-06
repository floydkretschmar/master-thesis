import os
import sys

import numpy as np

root_path = os.path.abspath(os.path.join('.'))
if root_path not in sys.path:
    sys.path.append(root_path)

from src.policy import ManualGradientPolicy


class ThetaDifferentiableFunction:
    def __init__(self, function):
        super().__init__()
        self.function = function

    def __call__(self, **function_args):
        return self.function(**function_args)

    def gradient(self, **gradient_args):
        raise NotImplementedError("Subclass must override gradient_wrt_model_parameters(self, **gradient_args).")


class UtilityFunction(ThetaDifferentiableFunction):
    def __init__(self, utility_function):
        super().__init__(utility_function)

    def _utility(self, x, s, y, decisions, ips_weights=None):
        utility = self.function(x=x, s=s, y=y, decisions=decisions)
        if ips_weights is not None:
            utility *= ips_weights
        return utility

    def __call__(self, x, s, y, decisions, ips_weights=None):
        return self._utility(x, s, y, decisions, ips_weights).mean(axis=0)

    def gradient(self, policy, x, s, y, decisions, ips_weights=None):
        utility = self(x=x, s=s, y=y, decisions=decisions, ips_weights=ips_weights)

        log_policy_gradient = policy.log_policy_gradient(x, s)
        utility_grad = log_policy_gradient * utility
        return np.mean(utility_grad, axis=0)


class FairnessFunction(ThetaDifferentiableFunction):
    def __init__(self, fairness_function, fairness_gradient_function):
        super().__init__(fairness_function)
        self.fairness_gradient_function = fairness_gradient_function

    def __call__(self, x, s, y, decisions, ips_weights=None):
        fairness = self.function(x=x, s=s, y=y, decisions=decisions, ips_weights=ips_weights)
        return fairness

    def gradient(self, policy, x, s, y, decisions, ips_weights=None):
        fairness_grad = self.fairness_gradient_function(x=x, s=s, y=y, decisions=decisions, ips_weights=ips_weights,
                                                        policy=policy)
        return fairness_grad


class OptimizationTarget:
    def __init__(self, initial_fairness_rate, utility_function, fairness_function):
        super().__init__()
        self.fairness_rate = initial_fairness_rate
        self.utility_function = utility_function
        self.fairness_function = fairness_function

    def __call__(self, x, s, y, decisions, ips_weights=None):
        raise NotImplementedError("Subclass must override __call__(self, utility, fairness).")

    @staticmethod
    def _parameter_dictionary(x, s, y, decisions, ips_weights):
        return {
            "x": x,
            "s": s,
            "y": y,
            "decisions": decisions,
            "ips_weights": ips_weights
        }

    def model_parameter_gradient(self, policy, x, s, y, decisions, ips_weights=None):
        raise NotImplementedError("Subclass must override gradient_wrt_model_parameters(self, **gradient_args).")

    def fairness_parameter_gradient(self, policy, x, s, y, decisions, ips_weights=None):
        raise NotImplementedError("Subclass must override gradient_wrt_model_parameters(self, **gradient_args).")


class PenaltyOptimizationTarget(OptimizationTarget):
    def __init__(self, initial_fairness_rate, utility_function, fairness_function):
        super().__init__(initial_fairness_rate, utility_function, fairness_function)

    def __call__(self, x, s, y, decisions, ips_weights=None):
        parameters = OptimizationTarget._parameter_dictionary(x, s, y, decisions, ips_weights)
        return self.utility_function(**parameters) - (self.fairness_rate / 2) * self.fairness_function(
            **parameters) ** 2

    def model_parameter_gradient(self, policy, x, s, y, decisions, ips_weights=None):
        assert isinstance(self.utility_function, ThetaDifferentiableFunction)
        assert isinstance(self.fairness_function, ThetaDifferentiableFunction)

        parameters = OptimizationTarget._parameter_dictionary(x, s, y, decisions, ips_weights)
        gradient = self.utility_function.gradient(policy=policy, **parameters)

        if self.fairness_rate > 0:
            fairness = self.fairness_function(**parameters)
            fairness_gradient = self.fairness_function.gradient(policy=policy, **parameters)

            grad_fairness = self.fairness_rate * fairness * fairness_gradient
            gradient -= grad_fairness

        return gradient

    def fairness_parameter_gradient(self, policy, x, s, y, decisions, ips_weights=None):
        parameters = OptimizationTarget._parameter_dictionary(x, s, y, decisions, ips_weights)
        return -(self.fairness_function(**parameters) ** 2) / 2


class LagrangianOptimizationTarget(OptimizationTarget):
    def __init__(self, initial_fairness_rate, utility_function, fairness_function):
        super().__init__(initial_fairness_rate, utility_function, fairness_function)

    def __call__(self, x, s, y, decisions, ips_weights=None):
        parameters = OptimizationTarget._parameter_dictionary(x, s, y, decisions, ips_weights)
        return self.utility_function(**parameters) - (self.fairness_rate * self.fairness_function(**parameters))

    def model_parameter_gradient(self, policy, x, s, y, decisions, ips_weights=None):
        assert isinstance(self.utility_function, ThetaDifferentiableFunction)
        assert isinstance(self.fairness_function, ThetaDifferentiableFunction)

        parameters = OptimizationTarget._parameter_dictionary(x, s, y, decisions, ips_weights)
        gradient = self.utility_function.gradient(policy=policy, **parameters)

        if self.fairness_rate > 0:
            fairness_gradient = self.fairness_function.gradient(policy=policy, **parameters)

            grad_fairness = self.fairness_rate * fairness_gradient
            gradient -= grad_fairness

        return gradient

    def fairness_parameter_gradient(self, policy, x, s, y, decisions, ips_weights=None):
        parameters = OptimizationTarget._parameter_dictionary(x, s, y, decisions, ips_weights)
        return -self.fairness_function(**parameters)


class Optimizer:
    def __init__(self, policy, optimization_target):
        super().__init__()
        self.optimization_target = optimization_target
        self.policy = policy

    def get_parameters(self):
        return {
            "theta": self.policy.get_model_parameters(),
            "lambda": self.optimization_target.fairness_rate
        }    

    def update_model_parameters(self, x, s, y, learning_rate, ips_weights=None):
        """ Updates the model parameters according to the specified update strategy.

            Args:
                x: The features of the n samples
                s: The sensitive attribute of the n samples
                y: The ground truth labels of the n samples
                learning_rate: The rate with which the fairness parameter will be updated.
                ips_weights: The weights used for inverse propensity scoring. If sampling_data=None 
                no IPS will be applied.
        """
        assert isinstance(self.policy, ManualGradientPolicy)

        # make decision according to current policy
        decisions = self.policy(x, s)

        # call the optimization target for gradient calculation
        gradient = self.optimization_target.model_parameter_gradient(self.policy, x, s, y, decisions, ips_weights)
        self.policy.update_model_parameters(gradient, learning_rate)

    def update_fairness_parameter(self, x, s, y, learning_rate, ips_weights=None):
        """ Updates the fairness parameter according to the specified update strategy.

            Args:
                x: The features of the n samples
                s: The sensitive attribute of the n samples
                y: The ground truth labels of the n samples
                learning_rate: The rate with which the fairness parameter will be updated.
                ips_weights: The weights used for inverse propensity scoring. If sampling_data=None 
                no IPS will be applied.
        """
        # make decision according to current policy
        decisions = self.policy(x, s)

        # call the optimization target for gradient calculation
        gradient = self.optimization_target.fairness_parameter_gradient(self.policy, x, s, y, decisions, ips_weights)
        self.optimization_target.fairness_rate -= learning_rate * gradient
