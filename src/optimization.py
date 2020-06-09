import abc
import os
import sys

import numpy as np
import torch
import torch.optim as optim
from copy import deepcopy

root_path = os.path.abspath(os.path.join('.'))
if root_path not in sys.path:
    sys.path.append(root_path)

from src.util import check_for_missing_kwargs, get_random, to_device


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


    def preprocess_data(self, x, s, y, ips_weights=None):
        raise NotImplementedError("Subclass must override preprocess_data(self, x, s, y, ips_weights=None).")

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

            if batch_end - batch_start < 2:
                break

            x_batch = x[batch_start:batch_end]
            s_batch = s[batch_start:batch_end]
            y_batch = y[batch_start:batch_end]

            if ips_weights is not None:
                ips_weight_batch = ips_weights[batch_start:batch_end]
            else:
                ips_weight_batch = None

            yield x_batch, s_batch, y_batch, ips_weight_batch

    def create_policy_checkpoint(self):
        raise NotImplementedError("Subclass must override create_policy_checkpoint(self)")

    def restore_last_policy_checkpoint(self):
        raise NotImplementedError("Subclass must override restore_last_policy_checkpoint(self)")

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
            gradient = self.optimization_target.fairness_parameter_gradient(policy=self.policy,
                                                                            x=x_batch,
                                                                            s=s_batch,
                                                                            y=y_batch,
                                                                            decisions=decisions_batch,
                                                                            decision_probabilities=decision_probability_batch,
                                                                            ips_weights=ips_weights_batch)
            self.optimization_target.fairness_rate += learning_rate * gradient

class PytorchStochasticGradientOptimizer(StochasticGradientOptimizer):
    def __init__(self, policy, optimization_target):
        super().__init__(policy, optimization_target)
        optimization_parameters, _ = self._policy.parameters
        self.model_optimizer = optim.SGD(optimization_parameters, lr=0.01)

    @property
    def parameters(self):
        params = super().parameters
        _, state_dict = self.policy.parameters
        params['theta'] = state_dict
        return params

    def _process_data(self, process_tensor_func, process_non_tensor_func, *data):
        return_data = []
        for array in data:
            if torch.is_tensor(array):
                return_data.append(process_tensor_func(array))
            else:
                return_data.append(process_non_tensor_func(array))

        if len(return_data) == 1:
            return return_data[0]
        else:
            return tuple(return_data)

    def create_policy_checkpoint(self):
        _, state_dictionary = self.policy.parameters
        self._state_dictionary = deepcopy(state_dictionary)

    def restore_last_policy_checkpoint(self):
        self.policy.parameters = deepcopy(self._state_dictionary)

    def preprocess_data(self, *data):
        def process_array(array):
            tensor = torch.from_numpy(array).float()
            return tensor

        return self._process_data(lambda array: array, process_array, *data)

    def postprocess_data(self, *data):
        return self._process_data(lambda array: array.detach().numpy(), lambda array: array, *data)

    def update_model_parameters(self, learning_rate, batch_size, x, s, y, ips_weights=None):
        """ Updates the model parameters using stochastic gradient descent.

            Args:
                learning_rate: The rate with which the model parameters will be updated.
                batch_size: The size of the batches into which the training data will be subdivided.
                x: The features of the n samples
                s: The sensitive attribute of the n samples
                y: The ground truth labels of the n samples
                ips_weights: The weights used for inverse propensity scoring. If ips_weights=None
                no IPS will be applied.
        """
        # update learning rate for SGD
        for param_group in self.model_optimizer.param_groups:
            param_group['lr'] = learning_rate

        # for each minibatch...
        for x_batch, s_batch, y_batch, ips_weights_batch in self._minibatch(batch_size, x, s, y,
                                                                            ips_weights):
            self.model_optimizer.zero_grad()
            decisions_batch, decision_probability_batch = self.policy(x_batch, s_batch)
            loss = self.optimization_target(policy=self.policy,
                                            x=x_batch,
                                            s=s_batch,
                                            y=y_batch,
                                            decisions=decisions_batch,
                                            decision_probabilities=decision_probability_batch,
                                            ips_weights=ips_weights_batch)
            loss.backward()
            self.model_optimizer.step()


class ManualStochasticGradientOptimizer(StochasticGradientOptimizer):
    """ The class for stochastic gradient methods of optimization with manual gradient updates. """

    def __init__(self, policy, optimization_target):
        assert isinstance(optimization_target, ManualGradientOptimizationTarget)
        super().__init__(policy, optimization_target)
        self.create_policy_checkpoint()

    def create_policy_checkpoint(self):
        self._theta = deepcopy(self.policy.parameters)

    def restore_last_policy_checkpoint(self):
        self.policy.parameters = deepcopy(self._theta)

    def preprocess_data(self, *data):
        if len(data) == 1:
            return data[0]

        return data

    def postprocess_data(self, *data):
        if len(data) == 1:
            return data[0]

        return data

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
            gradient = self.optimization_target.model_parameter_gradient(policy=self.policy,
                                                                         x=x_batch,
                                                                         s=s_batch,
                                                                         y=y_batch,
                                                                         decisions=decisions_batch,
                                                                         decision_probabilities=decision_probability_batch,
                                                                         ips_weights=ips_weights_batch)
            self.policy.parameters -= learning_rate * gradient


#################################### FUNCTION WRAPPERS FOR MANUAL GRADIENT #############################################


class DifferentiableFunction(abc.ABC):
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

    @abc.abstractmethod
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
    """ The class for a differentiable fairness function. """

    def __init__(self, fairness_function, fairness_gradient_function):
        """ Initializes a new FairnessFunction object.

        Args:
            fairness_function: The function that calculates the fariness value based on the given parameters.
            fairness_gradient_function: The function that calculates the fariness gradient based on the given parameters.
        """
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

class BaseOptimizationTarget(abc.ABC):
    """ The base class for a optimization target of an optimizer. """

    @abc.abstractmethod
    def __call__(self, **optimization_target_args):
        """ Returns the value of the optimization target with regards to the specified parameters.

        Args:
            optimization_target_args: The optimization target arguments.
        """
        pass


class ManualGradientOptimizationTarget(abc.ABC):
    """ The base class for a optimization target gradient. """

    @abc.abstractmethod
    def model_parameter_gradient(self, **optimization_target_args):
        """ Returns the gradient of the optimization target with regards to the policy parameters.

        Args:
            optimization_target_args: The optimization target arguments.
        """
        pass

    @abc.abstractmethod
    def fairness_parameter_gradient(self, **optimization_target_args):
        """ Returns the value of the optimization target with regards to the fairness value.

        Args:
            optimization_target_args: The optimization target arguments.
        """
        pass


class DualOptimizationTarget(BaseOptimizationTarget, abc.ABC):
    """ The base class for a optimization target that combines a utility function with a fairness constraint. """

    def __init__(self, fairness_rate, utility_function, fairness_function):
        super().__init__()
        self._fairness_rate = fairness_rate
        self._utility_function = utility_function
        self._fairness_function = fairness_function

    @property
    def fairness_function(self):
        """ Returns the fairness function of the combined optimization target. """
        return self._fairness_function

    @property
    def fairness_rate(self):
        """ Returns the fairness rate with which the fairness function is weighted. """
        return self._fairness_rate

    @fairness_rate.setter
    def fairness_rate(self, value):
        """ Sets the fairness rate with which the fairness function is weighted. """
        self._fairness_rate = value

    @property
    def utility_function(self):
        """ Returns the utility function of the combined optimization target. """
        return self._utility_function


class PenaltyOptimizationTarget(DualOptimizationTarget):
    """ The penalty optimization target that conbines utility and fairness constraint via a penalty term and is
     differentiable with regards to the model parameters."""

    def __init__(self, fairness_rate, utility_function, fairness_function):
        DualOptimizationTarget.__init__(self, fairness_rate, utility_function, fairness_function)

    def __call__(self, **optimization_target_args):
        """ Returns the gradient of the penalty optimization target with regards to the policy parameters.

        Args:
            optimization_target_args: The optimization target arguments.
        """
        return -self.utility_function(**optimization_target_args) + (self.fairness_rate / 2) * self.fairness_function(
            **optimization_target_args) ** 2


class ManualGradientPenaltyOptimizationTarget(PenaltyOptimizationTarget, ManualGradientOptimizationTarget):
    def __init__(self, fairness_rate, utility_function, fairness_function, fairness_gradient_function):
        super().__init__(fairness_rate,
                         UtilityFunction(utility_function),
                         FairnessFunction(fairness_function, fairness_gradient_function))

    def model_parameter_gradient(self, **optimization_target_args):
        """ Returns the value of the optimization target with regards to the specified parameters.

        Args:
            optimization_target_args: The optimization target arguments.
        """
        gradient = -self.utility_function.gradient(**optimization_target_args)

        if self.fairness_rate > 0:
            fairness = self.fairness_function(**optimization_target_args)
            fairness_gradient = self.fairness_function.gradient(**optimization_target_args)

            grad_fairness = self.fairness_rate * fairness * fairness_gradient
            gradient += grad_fairness

        return gradient

    def fairness_parameter_gradient(self, **optimization_target_args):
        raise TypeError("PenaltyOptimizationTarget does not support training of the fairness parameter.")


class LagrangianOptimizationTarget(DualOptimizationTarget):
    """ The lagrangian optimization target that conbines utility and fairness constraint via a lagrangien multiplier term
    and is differentiable with regards to the model parameters as well as the lagrangian multiplier"""

    def __init__(self, fairness_rate, utility_function, fairness_function):
        super().__init__(fairness_rate, utility_function, fairness_function)

    def __call__(self, **optimization_target_args):
        return -self.utility_function(**optimization_target_args) + self.fairness_rate * self.fairness_function(
            **optimization_target_args)


class ManualGradientLagrangianOptimizationTarget(LagrangianOptimizationTarget, ManualGradientOptimizationTarget):
    def __init__(self, fairness_rate, utility_function, fairness_function, fairness_gradient_function):
        super().__init__(fairness_rate,
                         UtilityFunction(utility_function),
                         FairnessFunction(fairness_function, fairness_gradient_function))

    def model_parameter_gradient(self, **optimization_target_args):
        """ Returns the gradient of the lagrangian optimization target with regards to the policy parameters.

        Args:
            optimization_target_args: The optimization target arguments.
        """
        gradient = -self.utility_function.gradient(**optimization_target_args)

        fairness_gradient = self.fairness_function.gradient(**optimization_target_args)
        grad_fairness = self.fairness_rate * fairness_gradient

        gradient += grad_fairness

        return gradient

    def fairness_parameter_gradient(self, **optimization_target_args):
        """ Returns the value of the lagrangian optimization target with regards to the fairness value.

        Args:
            optimization_target_args: The optimization target arguments.
        """
        return self.fairness_function(**optimization_target_args)


class AugmentedLagrangianOptimizationTarget(LagrangianOptimizationTarget):
    """ The augmented lagrangian optimization target that conbines utility and fairness constraint via a lagrangien
    multiplier term as well as a penalty term and is differentiable with regards to the model parameters as well as
    the lagrangian multiplier"""

    def __init__(self, fairness_rate, penalty_constant, utility_function, fairness_function):
        super().__init__(fairness_rate, utility_function, fairness_function)
        self.penalty_constant = penalty_constant

    def __call__(self, **optimization_target_args):
        lagrangian_result = super().__call__(**optimization_target_args)
        return lagrangian_result + (self.penalty_constant / 2) * self.fairness_function(**optimization_target_args) ** 2


# WATCH OUT: MRO = ManualGradientAugmentedLagrangianOptimizationTarget
# -> AugmentedLagrangianOptimizationTarget for super().__call__
# -> ManualGradientLagrangianOptimizationTarget for super().model_parameter_gradient
class ManualGradientAugmentedLagrangianOptimizationTarget(AugmentedLagrangianOptimizationTarget,
                                                          ManualGradientLagrangianOptimizationTarget):
    def __init__(self, fairness_rate, penalty_constant, utility_function, fairness_function, fairness_gradient_function):
        super().__init__(fairness_rate,
                         penalty_constant,
                         UtilityFunction(utility_function),
                         FairnessFunction(fairness_function, fairness_gradient_function))

    def model_parameter_gradient(self, **optimization_target_args):
        # lagrangian gradient
        gradient = super().model_parameter_gradient(**optimization_target_args)

        # augmentation
        fairness = self.fairness_function(**optimization_target_args)
        fairness_gradient = self.fairness_function.gradient(**optimization_target_args)
        grad_augmented = self.penalty_constant * fairness * fairness_gradient
        gradient += grad_augmented

        return gradient
