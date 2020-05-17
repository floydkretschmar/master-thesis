import os
import sys

import numpy as np

root_path = os.path.abspath(os.path.join('.'))
if root_path not in sys.path:
    sys.path.append(root_path)

from src.util import check_for_missing_kwargs, get_random


########################################$ OPTIMIZERS ####################################################

# TODO: Add Pytorch supporting optimizer

class StochasticGradientOptimizer:
    """ The base class for stochastic gradient methods of optimization. """

    def __init__(self, policy, optimization_target):
        super().__init__()
        self._optimization_target = optimization_target
        self._policy = policy

    @property
    def optimization_target(self):
        """ Returns the optimization target according to which the optimizer optimizes the policy. """
        return self._optimization_target

    @property
    def parameters(self):
        """ Returns the paramters being optimized. """
        return {
            "theta": self.policy.parameters,
            "lambda": self.optimization_target.fairness_rate
        }

    @property
    def policy(self):
        """ Returns the policy being optimized. """
        return self._policy

    def _minibatch(self, batch_size, x, s, y, ips_weights=None):
        """ Creates minibatches for stochastic gradient ascent according to the epochs and
        batch size.

        Args:
            data: The training data
            batch_size: The minibatch size of SGD.
        """
        # minibatching
        indices = get_random().permutation(x.shape[0])
        for batch_start in range(0, len(indices), batch_size):
            batch_end = min(batch_start + batch_size, len(indices))

            x_batch = x[batch_start:batch_end]
            s_batch = s[batch_start:batch_end]
            y_batch = y[batch_start:batch_end]

            if ips_weights is not None:
                ips_weight_batch = ips_weights[batch_start:batch_end]
            else:
                ips_weight_batch = None

            yield x_batch, s_batch, y_batch, ips_weight_batch

    def update_model_parameters(self, learning_rate, batch_size, x, s, y, ips_weights=None):
        """ Updates the model parameters according to the specified update strategy.

            Args:
                learning_rate: The rate with which the model parameters will be updated.
                batch_size: The size of the batches into which the training data will be subdivided.
                x: The features of the n samples
                s: The sensitive attribute of the n samples
                y: The ground truth labels of the n samples
                ips_weights: The weights used for inverse propensity scoring. If ips_weights=None
                no IPS will be applied.
        """
        raise NotImplementedError(
            "Subclass must override update_model_parameters(self, x, s, y, learning_rate, ips_weights=None)")

    def update_fairness_parameter(self, learning_rate, batch_size, x, s, y, ips_weights=None):
        """ Updates the fairness parameter according to the specified update strategy.

            Args:
                learning_rate: The rate with which the fairness parameter will be updated.
                batch_size: The size of the batches into which the training data will be subdivided.
                x: The features of the n samples
                s: The sensitive attribute of the n samples
                y: The ground truth labels of the n samples
                ips_weights: The weights used for inverse propensity scoring. If ips_weights=None
                no IPS will be applied.
        """
        raise NotImplementedError(
            "Subclass must override update_fairness_parameter(self, x, s, y, learning_rate, ips_weights=None)")


class ManualStochasticGradientOptimizer(StochasticGradientOptimizer):
    """ The class for stochastic gradient methods of optimization with manual gradient updates. """

    def __init__(self, policy, optimization_target):
        assert isinstance(optimization_target, DifferentiableOptimizationTarget)
        super().__init__(policy, optimization_target)

    def update_model_parameters(self, learning_rate, batch_size, x, s, y, ips_weights=None):
        """ Manually updates the model parameters using stochastic gradient descent.

            Args:
                learning_rate: The rate with which the model parameters will be updated.
                batch_size: The size of the batches into which the training data will be subdivided.
                x: The features of the n samples
                s: The sensitive attribute of the n samples
                y: The ground truth labels of the n samples
                ips_weights: The weights used for inverse propensity scoring. If ips_weights=None
                no IPS will be applied.
        """
        # for each minibatch...
        for x_batch, s_batch, y_batch, ips_weights_batch in self._minibatch(batch_size, x, s, y, ips_weights):
            # make decision according to current policy
            decisions_batch, decision_probability_batch = self.policy(x_batch, s_batch)

            # call the optimization target for gradient calculation
            gradient = self.optimization_target.model_parameter_gradient(self.policy,
                                                                         x_batch,
                                                                         s_batch,
                                                                         y_batch,
                                                                         decisions_batch,
                                                                         decision_probability_batch,
                                                                         ips_weights_batch)
            self.policy.theta -= learning_rate * gradient

    def update_fairness_parameter(self, learning_rate, batch_size, x, s, y, ips_weights=None):
        """ Manually updates the fairness parameter using stochastic gradient descent.

            Args:
                learning_rate: The rate with which the model parameters will be updated.
                batch_size: The size of the batches into which the training data will be subdivided.
                x: The features of the n samples
                s: The sensitive attribute of the n samples
                y: The ground truth labels of the n samples
                ips_weights: The weights used for inverse propensity scoring. If ips_weights=None
                no IPS will be applied.
        """
        # for each minibatch...
        for x_batch, s_batch, y_batch, ips_weights_batch in self._minibatch(batch_size, x, s, y, ips_weights):
            # make decision according to current policy
            decisions_batch, decision_probability_batch = self.policy(x_batch, s_batch)

            # call the optimization target for gradient calculation
            gradient = self.optimization_target.fairness_parameter_gradient(self.policy,
                                                                            x_batch,
                                                                            s_batch,
                                                                            y_batch,
                                                                            decisions_batch,
                                                                            decision_probability_batch,
                                                                            ips_weights_batch)
            self.optimization_target.fairness_rate += learning_rate * gradient


#################################### FUNCTION WRAPPERS FOR MANUAL GRADIENT #############################################


class DifferentiableFunction:
    """ The base class for a differentiable function. """

    def __init__(self, function):
        super().__init__()
        self.function = function

    def __call__(self, **function_args):
        """ Returns the value of the function according to the specified parameters.

        Args:
            function_args: The function arguments.

        Returns:
            result: The function result.
        """
        return self.function(**function_args)

    def gradient(self, **function_args):
        """ Returns the gradient of the function according to the specified parameters.

        Args:
            function_args: The function arguments.

        Returns:
            result: The function result.
        """
        raise NotImplementedError("Subclass must override gradient(self, **function_args).")


class UtilityFunction(DifferentiableFunction):
    """ The class for a differentiable utility function. """

    def __init__(self, utility_function):
        """ Initializes a new UtilityFunction object.

        Args:
            utility_function: The function that calculates utility based on the given parameters.
        """
        super().__init__(utility_function)

    def _utility(self, **function_args):
        """ Calculates utility using the underlying function and applies IPS if weights are specified.

        Args:
            function_args: The utility function arguments.
        """
        utility = self.function(**function_args)
        if function_args["ips_weights"] is not None:
            utility *= function_args["ips_weights"]
        return utility

    def __call__(self, **function_args):
        """ Returns the utility for the specified arguments.

        Args:
            function_args: The utility function arguments.
        """
        function_args["ips_weights"] = function_args["ips_weights"] if "ips_weights" in function_args else None
        return self._utility(**function_args).mean(axis=0)

    def gradient(self, **function_args):
        """ Returns the utility gradient for the specified arguments.

        Args:
            function_args: The utility function arguments.
        """
        check_for_missing_kwargs("UtilityFunction.gradient", ["policy", "x", "s"], function_args)
        function_args["ips_weights"] = function_args["ips_weights"] if "ips_weights" in function_args else None
        utility = self._utility(**function_args)

        log_policy_gradient = function_args["policy"].log_policy_gradient(function_args["x"], function_args["s"])
        utility_grad = log_policy_gradient * utility
        return np.mean(utility_grad, axis=0)


class FairnessFunction(DifferentiableFunction):
    def __init__(self, fairness_function, fairness_gradient_function):
        super().__init__(fairness_function)
        self.fairness_gradient_function = fairness_gradient_function

    def __call__(self, **function_args):
        """ Returns the fairness value for the specified arguments.

        Args:
            function_args: The fairness function arguments.
        """
        function_args["ips_weights"] = function_args["ips_weights"] if "ips_weights" in function_args else None
        fairness = self.function(**function_args)
        return fairness

    def gradient(self, **function_args):
        """ Returns the fairness gradient for the specified arguments.

        Args:
            function_args: The fairness function arguments.
        """
        function_args["ips_weights"] = function_args["ips_weights"] if "ips_weights" in function_args else None
        fairness_grad = self.fairness_gradient_function(**function_args)
        return fairness_grad


########################################$ OPTIMIZATION TARGETS ####################################################

class BaseOptimizationTarget:
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
    def build(initial_fairness_rate, utility_function, fairness_function):
        raise NotImplementedError(
            "Subclass must override build(initial_fairness_rate, utility_function, fairness_function).")

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


class DifferentiableOptimizationTarget(BaseOptimizationTarget):
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


class PenaltyOptimizationTarget(DifferentiableOptimizationTarget):
    class _PenaltyOptimizationTarget(BaseOptimizationTarget):
        def __init__(self, initial_fairness_rate, utility_function, fairness_function):
            super().__init__(initial_fairness_rate, utility_function, fairness_function)

        def __call__(self, policy, x, s, y, decisions, decision_probabilities, ips_weights=None):
            parameters = BaseOptimizationTarget._parameter_dictionary(x, s, y, decisions, decision_probabilities,
                                                                      ips_weights,
                                                                      policy)
            return -self.utility_function(**parameters) + (self.fairness_rate / 2) * self.fairness_function(
                **parameters) ** 2

    def __init__(self, fairness_constant, utility_function, fairness_function):
        assert isinstance(utility_function, DifferentiableFunction)
        assert isinstance(fairness_function, DifferentiableFunction)
        super().__init__(self._PenaltyOptimizationTarget(fairness_constant, utility_function, fairness_function))

    def model_parameter_gradient(self, policy, x, s, y, decisions, decision_probabilities, ips_weights=None):
        parameters = BaseOptimizationTarget._parameter_dictionary(x, s, y, decisions, decision_probabilities,
                                                                  ips_weights,
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

    @staticmethod
    def build(initial_fairness_rate, utility_function, fairness_function):
        if isinstance(utility_function, DifferentiableFunction) and isinstance(fairness_function,
                                                                               DifferentiableFunction):
            return PenaltyOptimizationTarget(initial_fairness_rate, utility_function, fairness_function)
        else:
            return PenaltyOptimizationTarget._PenaltyOptimizationTarget(initial_fairness_rate, utility_function,
                                                                        fairness_function)


class LagrangianOptimizationTarget(DifferentiableOptimizationTarget):
    class _LagrangianOptimizationTarget(BaseOptimizationTarget):
        def __init__(self, initial_lambda, utility_function, fairness_function):
            super().__init__(initial_lambda, utility_function, fairness_function)

        def __call__(self, policy, x, s, y, decisions, decision_probabilities, ips_weights=None):
            parameters = BaseOptimizationTarget._parameter_dictionary(x, s, y, decisions, decision_probabilities,
                                                                      ips_weights,
                                                                      policy)
            return -self.utility_function(**parameters) + self.fairness_rate * self.fairness_function(**parameters)

    def __init__(self, initial_lambda, utility_function, fairness_function):
        assert isinstance(utility_function, DifferentiableFunction)
        assert isinstance(fairness_function, DifferentiableFunction)
        super().__init__(self._LagrangianOptimizationTarget(initial_lambda, utility_function, fairness_function))

    def model_parameter_gradient(self, policy, x, s, y, decisions, decision_probabilities, ips_weights=None):
        parameters = BaseOptimizationTarget._parameter_dictionary(x, s, y, decisions, decision_probabilities,
                                                                  ips_weights,
                                                                  policy)
        gradient = -self.utility_function.gradient(**parameters)

        fairness_gradient = self.fairness_function.gradient(**parameters)
        grad_fairness = self.fairness_rate * fairness_gradient

        gradient += grad_fairness

        return gradient

    def fairness_parameter_gradient(self, policy, x, s, y, decisions, decision_probabilities, ips_weights=None):
        parameters = BaseOptimizationTarget._parameter_dictionary(x, s, y, decisions, decision_probabilities,
                                                                  ips_weights,
                                                                  policy)
        return self.fairness_function(**parameters)

    @staticmethod
    def build(initial_fairness_rate, utility_function, fairness_function):
        if isinstance(utility_function, DifferentiableFunction) and isinstance(fairness_function,
                                                                               DifferentiableFunction):
            return LagrangianOptimizationTarget(initial_fairness_rate, utility_function, fairness_function)
        else:
            return LagrangianOptimizationTarget._LagrangianOptimizationTarget(initial_fairness_rate,
                                                                              utility_function,
                                                                              fairness_function)


class AugmentedLagrangianOptimizationTarget(LagrangianOptimizationTarget):
    class _AugmentedLagrangianOptimizationTarget(LagrangianOptimizationTarget._LagrangianOptimizationTarget):
        def __init__(self, initial_lambda, utility_function, fairness_function, penalty_constant):
            super().__init__(initial_lambda, utility_function, fairness_function)
            self.penalty_constant = penalty_constant

        def __call__(self, policy, x, s, y, decisions, decision_probabilities, ips_weights=None):
            parameters = BaseOptimizationTarget._parameter_dictionary(x, s, y, decisions, decision_probabilities,
                                                                      ips_weights,
                                                                      policy)
            lagrangian_result = super.__call__(policy, x, s, y, decisions, decision_probabilities, ips_weights)

            return lagrangian_result + self.penalty_constant * self.fairness_function(**parameters)

    def __init__(self, initial_lambda, utility_function, fairness_function, penalty_constant):
        super().__init__(initial_lambda, utility_function, fairness_function)
        self.optimization_target = self._AugmentedLagrangianOptimizationTarget(initial_lambda,
                                                                               utility_function,
                                                                               fairness_function,
                                                                               penalty_constant)

    def model_parameter_gradient(self, policy, x, s, y, decisions, decision_probabilities, ips_weights=None):
        parameters = BaseOptimizationTarget._parameter_dictionary(x, s, y, decisions, decision_probabilities,
                                                                  ips_weights,
                                                                  policy)
        # lagrangian gradient
        gradient = super().model_parameter_gradient(policy, x, s, y, decisions, decision_probabilities, ips_weights)

        # augmentation
        fairness = self.fairness_function(**parameters)
        fairness_gradient = self.fairness_function.gradient(**parameters)
        grad_augmented = self.penalty_constant * fairness * fairness_gradient
        gradient += grad_augmented

        return gradient

    def fairness_parameter_gradient(self, policy, x, s, y, decisions, decision_probabilities, ips_weights=None):
        parameters = BaseOptimizationTarget._parameter_dictionary(x, s, y, decisions, decision_probabilities,
                                                                  ips_weights,
                                                                  policy)
        return self.fairness_function(**parameters)

    @staticmethod
    def build(initial_fairness_rate, utility_function, fairness_function):
        if isinstance(utility_function, DifferentiableFunction) and isinstance(fairness_function,
                                                                               DifferentiableFunction):
            return AugmentedLagrangianOptimizationTarget(initial_fairness_rate, utility_function, fairness_function)
        else:
            return AugmentedLagrangianOptimizationTarget._AugmentedLagrangianOptimizationTarget(initial_fairness_rate,
                                                                                                utility_function,
                                                                                                fairness_function)
