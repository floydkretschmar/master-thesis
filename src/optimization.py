import os
import sys

import numpy as np

root_path = os.path.abspath(os.path.join('.'))
if root_path not in sys.path:
    sys.path.append(root_path)

from src.util import check_for_missing_kwargs


########################################$ BASE OPTIMIZATION CLASSES ####################################################

class OptimizationTarget:
    def __init__(self, initial_fairness_rate, utility_function, fairness_function):
        super().__init__()
        self._fairness_rate = initial_fairness_rate
        self._utility_function = utility_function
        self._fairness_function = fairness_function

    def __call__(self, policy, x, s, y, decisions, decision_probabilities, ips_weights=None):
        raise NotImplementedError("Subclass must override __call__(self, utility, fairness).")

    @property
    def fairness_rate(self):
        return self._fairness_rate

    @fairness_rate.setter
    def fairness_rate(self, value):
        self._fairness_rate = value

    @property
    def utility_function(self):
        return self._utility_function

    @property
    def fairness_function(self):
        return self._fairness_function

    @staticmethod
    def _parameter_dictionary(x, s, y, decisions, decision_probabilities, ips_weights, policy):
        return {
            "x": x,
            "s": s,
            "y": y,
            "decisions": decisions,
            "decision_probabilities": decision_probabilities,
            "ips_weights": ips_weights,
            "policy": policy
        }


class LagrangianOptimizationTarget(OptimizationTarget):
    def __init__(self, initial_lambda, utility_function, fairness_function):
        super().__init__(initial_lambda, utility_function, fairness_function)

    def __call__(self, policy, x, s, y, decisions, decision_probabilities, ips_weights=None):
        parameters = OptimizationTarget._parameter_dictionary(x, s, y, decisions, decision_probabilities, ips_weights,
                                                              policy)
        return -self.utility_function(**parameters) + self.fairness_rate * self.fairness_function(**parameters)


class PenaltyOptimizationTarget(OptimizationTarget):
    def __init__(self, initial_fairness_rate, utility_function, fairness_function):
        super().__init__(initial_fairness_rate, utility_function, fairness_function)

    def __call__(self, policy, x, s, y, decisions, decision_probabilities, ips_weights=None):
        parameters = OptimizationTarget._parameter_dictionary(x, s, y, decisions, decision_probabilities, ips_weights,
                                                              policy)
        return -self.utility_function(**parameters) + (self.fairness_rate / 2) * self.fairness_function(
            **parameters) ** 2


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
        raise NotImplementedError(
            "Subclass must override update_model_parameters(self, x, s, y, learning_rate, ips_weights=None)")

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
        raise NotImplementedError(
            "Subclass must override update_fairness_parameter(self, x, s, y, learning_rate, ips_weights=None)")


############################################ MANUAL DIFFERENTIATION ####################################################


class DifferentiableFunction:
    def __init__(self, function):
        super().__init__()
        self.function = function

    def __call__(self, **function_args):
        return self.function(**function_args)

    def gradient(self, **gradient_args):
        raise NotImplementedError("Subclass must override gradient_wrt_model_parameters(self, **gradient_args).")


class UtilityFunction(DifferentiableFunction):
    def __init__(self, utility_function):
        super().__init__(utility_function)

    def _utility(self, policy, x, s, y, decisions, decision_probabilities, ips_weights=None):
        utility = self.function(x=x, s=s, y=y, decisions=decisions, decision_probabilities=decision_probabilities,
                                policy=policy)
        if ips_weights is not None:
            utility *= ips_weights
        return utility

    def __call__(self, **function_args):
        check_for_missing_kwargs("UtilityFunction()", ["x", "s", "y", "decisions", "decision_probabilities", "policy"],
                                 function_args)
        return self._utility(**function_args).mean(axis=0)

    def gradient(self, **gradient_args):
        check_for_missing_kwargs("UtilityFunction()", ["x", "s", "y", "decisions", "decision_probabilities", "policy"],
                                 gradient_args)
        utility = self._utility(**gradient_args)

        log_policy_gradient = gradient_args["policy"].log_policy_gradient(gradient_args["x"], gradient_args["s"])
        utility_grad = log_policy_gradient * utility
        return np.mean(utility_grad, axis=0)


class FairnessFunction(DifferentiableFunction):
    def __init__(self, fairness_function, fairness_gradient_function):
        super().__init__(fairness_function)
        self.fairness_gradient_function = fairness_gradient_function

    def __call__(self, **function_args):
        check_for_missing_kwargs("FairnessFunction()", ["x", "s", "y", "decisions", "decision_probabilities", "policy"],
                                 function_args)
        fairness = self.function(**function_args)
        return fairness

    def gradient(self, **gradient_args):
        check_for_missing_kwargs("FairnessFunction()", ["x", "s", "y", "decisions", "decision_probabilities", "policy"],
                                 gradient_args)
        fairness_grad = self.fairness_gradient_function(**gradient_args)
        return fairness_grad


class DifferentiableOptimizationTarget(OptimizationTarget):
    def __init__(self, optimization_target):
        self.optimization_target = optimization_target

    def __call__(self, policy, x, s, y, decisions, decision_probabilities, ips_weights=None):
        return self.optimization_target(policy, x, s, y, decisions, decision_probabilities, ips_weights)

    @property
    def fairness_rate(self):
        return self.optimization_target.fairness_rate

    @fairness_rate.setter
    def fairness_rate(self, value):
        self.optimization_target.fairness_rate = value

    @property
    def utility_function(self):
        return self.optimization_target.utility_function

    @property
    def fairness_function(self):
        return self.optimization_target.fairness_function

    def model_parameter_gradient(self, policy, x, s, y, decisions, decision_probabilities, ips_weights=None):
        raise NotImplementedError("Subclass must override gradient_wrt_model_parameters(self, **gradient_args).")

    def fairness_parameter_gradient(self, policy, x, s, y, decisions, decision_probabilities, ips_weights=None):
        raise NotImplementedError("Subclass must override gradient_wrt_model_parameters(self, **gradient_args).")


class DifferentiablePenaltyOptimizationTarget(DifferentiableOptimizationTarget):
    def __init__(self, fairness_constant, utility_function, fairness_function):
        assert isinstance(utility_function, DifferentiableFunction)
        assert isinstance(fairness_function, DifferentiableFunction)
        super().__init__(PenaltyOptimizationTarget(fairness_constant, utility_function, fairness_function))

    def model_parameter_gradient(self, policy, x, s, y, decisions, decision_probabilities, ips_weights=None):
        parameters = OptimizationTarget._parameter_dictionary(x, s, y, decisions, decision_probabilities, ips_weights,
                                                              policy)
        gradient = -self.utility_function.gradient(**parameters)

        if self.fairness_rate > 0:
            fairness = self.fairness_function(**parameters)
            fairness_gradient = self.fairness_function.gradient(**parameters)

            grad_fairness = self.fairness_rate * fairness * fairness_gradient
            gradient += grad_fairness

        return gradient

    def fairness_parameter_gradient(self, policy, x, s, y, decisions, decision_probabilities, ips_weights=None):
        raise TypeError("Fairness gradient not supported for PenaltyOptimizationTarget")


class DifferentiableLagrangianOptimizationTarget(DifferentiableOptimizationTarget):
    def __init__(self, initial_lambda, utility_function, fairness_function):
        assert isinstance(utility_function, DifferentiableFunction)
        assert isinstance(fairness_function, DifferentiableFunction)
        super().__init__(LagrangianOptimizationTarget(initial_lambda, utility_function, fairness_function))

    def model_parameter_gradient(self, policy, x, s, y, decisions, decision_probabilities, ips_weights=None):
        parameters = OptimizationTarget._parameter_dictionary(x, s, y, decisions, decision_probabilities, ips_weights,
                                                              policy)
        gradient = -self.utility_function.gradient(**parameters)

        fairness_gradient = self.fairness_function.gradient(**parameters)
        grad_fairness = self.fairness_rate * fairness_gradient

        gradient += grad_fairness

        return gradient

    def fairness_parameter_gradient(self, policy, x, s, y, decisions, decision_probabilities, ips_weights=None):
        parameters = OptimizationTarget._parameter_dictionary(x, s, y, decisions, decision_probabilities, ips_weights,
                                                              policy)
        return self.fairness_function(**parameters)


class ManualGradientOptimizer(Optimizer):
    def __init__(self, policy, optimization_target):
        assert isinstance(optimization_target, DifferentiableOptimizationTarget)
        super().__init__(policy, optimization_target)

    def update_model_parameters(self, x, s, y, learning_rate, ips_weights=None):
        # make decision according to current policy
        decisions, decision_probability = self.policy(x, s)

        # call the optimization target for gradient calculation
        gradient = self.optimization_target.model_parameter_gradient(self.policy, x, s, y, decisions,
                                                                     decision_probability, ips_weights)
        self.policy.theta -= learning_rate * gradient

    def update_fairness_parameter(self, x, s, y, learning_rate, ips_weights=None):
        # make decision according to current policy
        decisions, decision_probability = self.policy(x, s)

        # call the optimization target for gradient calculation
        gradient = self.optimization_target.fairness_parameter_gradient(self.policy, x, s, y, decisions,
                                                                        decision_probability, ips_weights)
        self.optimization_target.fairness_rate += learning_rate * gradient
