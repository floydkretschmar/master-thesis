import abc
import os
import sys

import torch
import torch.optim as optim
import numbers
import numpy as np
from copy import deepcopy

root_path = os.path.abspath(os.path.join('.'))
if root_path not in sys.path:
    sys.path.append(root_path)

from src.util import check_for_missing_kwargs, get_random, to_device


########################################$ OPTIMIZERS ####################################################

# TODO: Add Pytorch supporting optimizer

class StochasticGradientOptimizer:
    """ The base class for stochastic gradient methods of optimization. """

    def __init__(self, policy, optimization_target, seed=None):
        super().__init__()
        self._optimization_target = optimization_target
        self._policy = policy
        self._seed = seed

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, value):
        self._seed = value

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
        # minibatching except for when batch size > num data points
        if batch_size >= x.shape[0]:
            yield x, s, y, ips_weights
        else:
            np_random, _ = get_random(self._seed)
            indices = np_random.permutation(x.shape[0])
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

            returns:
                average_gradient: The average gradient across all minibatches
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

            tmp = self.optimization_target.fairness_rate + learning_rate * gradient
            self.optimization_target.fairness_rate = tmp

class PytorchStochasticGradientOptimizer(StochasticGradientOptimizer):
    def __init__(self, policy, optimization_target, seed=None):
        super().__init__(policy, optimization_target, seed)
        optimization_parameters, _ = self._policy.parameters
        self.model_optimizer = optim.SGD(optimization_parameters, lr=0.01)
        with torch.no_grad():
            self.optimization_target.fairness_rate = torch.tensor(self.optimization_target.fairness_rate)

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
        for x_batch, s_batch, y_batch, _ in self._minibatch(batch_size, x, s, y, ips_weights):
            self.model_optimizer.zero_grad()
            decisions_batch, decision_probability_batch = self.policy(x_batch, s_batch)
            loss = self.optimization_target(policy=self.policy,
                                            x=x_batch,
                                            s=s_batch,
                                            y=y_batch,
                                            decisions=decisions_batch,
                                            decision_probabilities=decision_probability_batch)
            loss.backward(retain_graph=False)
            self.model_optimizer.step()

    def update_fairness_parameter(self, learning_rate, batch_size, x, s, y, ips_weights=None):
        with torch.no_grad():
            super().update_fairness_parameter(learning_rate, batch_size, x, s, y, ips_weights)


class ManualStochasticGradientOptimizer(StochasticGradientOptimizer):
    """ The class for stochastic gradient methods of optimization with manual gradient updates. """

    def __init__(self, policy, optimization_target, seed=None):
        assert isinstance(optimization_target, ManualGradientOptimizationTarget)
        super().__init__(policy, optimization_target, seed)
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


class DifferentiableFunction:
    """ The base class for a differentiable function. """

    def __init__(self, function, gradient_function):
        super().__init__()
        self._function = function
        self._gradient_function = gradient_function

    def __call__(self, **function_args):
        """ Returns the value of the function according to the specified parameters.

        Args:
            function_args: The function arguments.

        Returns:
            result: The function result.
        """
        function_args["ips_weights"] = function_args["ips_weights"] if "ips_weights" in function_args else None
        return self._function(**function_args)

    @abc.abstractmethod
    def gradient(self, **function_args):
        """ Returns the gradient of the function according to the specified parameters.

        Args:
            function_args: The function arguments.

        Returns:
            result: The function result.
        """
        function_args["ips_weights"] = function_args["ips_weights"] if "ips_weights" in function_args else None
        return self._gradient_function(**function_args)


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
        self._utility_function = utility_function
        self._fairness_function = fairness_function
        self._fairness_rate = fairness_rate

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

    def fairness_parameter_gradient(self, **optimization_target_args):
        raise TypeError("PenaltyOptimizationTarget does not support training of the fairness parameter.")


class ManualGradientPenaltyOptimizationTarget(PenaltyOptimizationTarget, ManualGradientOptimizationTarget):
    def __init__(self, fairness_rate,
                 utility_function,
                 utility_gradient_function,
                 fairness_function,
                 fairness_gradient_function):
        super().__init__(fairness_rate,
                         DifferentiableFunction(utility_function, utility_gradient_function),
                         DifferentiableFunction(fairness_function, fairness_gradient_function))

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


class LagrangianOptimizationTarget(DualOptimizationTarget):
    """ The lagrangian optimization target that conbines utility and fairness constraint via a lagrangien multiplier term
    and is differentiable with regards to the model parameters as well as the lagrangian multiplier"""

    def __init__(self, fairness_rate, utility_function, fairness_function, error_delta=0.0):
        super().__init__(fairness_rate, utility_function, fairness_function)
        self._error_delta = error_delta
        self.fairness_rate = fairness_rate

    def __call__(self, **optimization_target_args):
        if self.error_delta == 0.0:
            return -self.utility_function(**optimization_target_args) + self.fairness_rate * self.fairness_function(
                **optimization_target_args)
        else:
            lambda1, lambda2 = self.fairness_rate
            return -self.utility_function(**optimization_target_args) \
                   + lambda1 * (-self.fairness_function(**optimization_target_args) - self._error_delta) \
                   + lambda2 * (self.fairness_function(**optimization_target_args) - self._error_delta)

    @property
    def error_delta(self):
        """ Returns the fairness rate with which the fairness function is weighted. """
        return self._error_delta

    @property
    def fairness_rate(self):
        """ Returns the fairness rate with which the fairness function is weighted. """
        return self._fairness_rate

    @fairness_rate.setter
    def fairness_rate(self, value):
        """ Sets the fairness rate with which the fairness function is weighted. """
        if self.error_delta != 0.0 and isinstance(value, numbers.Number):
            self._fairness_rate = np.array([value, value])
        elif self.error_delta != 0.0:
            self._fairness_rate = np.array([max(value[0], 0), max(value[1], 0)])
        else:
            self._fairness_rate = value

    def fairness_parameter_gradient(self, **optimization_target_args):
        """ Returns the value of the lagrangian optimization target with regards to the fairness value.

        Args:
            optimization_target_args: The optimization target arguments.
        """
        if self.error_delta == 0.0:
            return self.fairness_function(**optimization_target_args)
        else:
            return np.array([-self.fairness_function(**optimization_target_args), self.fairness_function(**optimization_target_args)]).squeeze()


class ManualGradientLagrangianOptimizationTarget(LagrangianOptimizationTarget, ManualGradientOptimizationTarget):
    def __init__(self, fairness_rate,
                 utility_function,
                 utility_gradient_function,
                 fairness_function,
                 fairness_gradient_function,
                 error_delta=0.0):
        super().__init__(fairness_rate,
                         DifferentiableFunction(utility_function, utility_gradient_function),
                         DifferentiableFunction(fairness_function, fairness_gradient_function),
                         error_delta)

    def model_parameter_gradient(self, **optimization_target_args):
        """ Returns the gradient of the lagrangian optimization target with regards to the policy parameters.

        Args:
            optimization_target_args: The optimization target arguments.
        """
        gradient = -self.utility_function.gradient(**optimization_target_args)
        fairness_gradient = self.fairness_function.gradient(**optimization_target_args)

        if self.error_delta == 0.0:
            grad_fairness = self.fairness_rate * fairness_gradient
        else:
            lambda1, lambda2 = self.fairness_rate
            grad_fairness = lambda1 * -fairness_gradient + lambda2 * fairness_gradient

        gradient += grad_fairness
        return gradient


class AugmentedLagrangianOptimizationTarget(LagrangianOptimizationTarget):
    """ The augmented lagrangian optimization target that conbines utility and fairness constraint via a lagrangien
    multiplier term as well as a penalty term and is differentiable with regards to the model parameters as well as
    the lagrangian multiplier"""

    def __init__(self, fairness_rate, utility_function, fairness_function, penalty_constant):
        LagrangianOptimizationTarget.__init__(self, fairness_rate, utility_function, fairness_function)
        self.penalty_constant = penalty_constant

    def __call__(self, **optimization_target_args):
        lagrangian_result = super().__call__(**optimization_target_args)
        return lagrangian_result + (self.penalty_constant / 2) * self.fairness_function(**optimization_target_args) ** 2


# WATCH OUT: MRO = ManualGradientAugmentedLagrangianOptimizationTarget
# -> AugmentedLagrangianOptimizationTarget for super().__call__
# -> ManualGradientLagrangianOptimizationTarget for super().model_parameter_gradient
class ManualGradientAugmentedLagrangianOptimizationTarget(AugmentedLagrangianOptimizationTarget,
                                                          ManualGradientLagrangianOptimizationTarget):
    def __init__(self, fairness_rate,
                 utility_function,
                 utility_gradient_function,
                 fairness_function,
                 fairness_gradient_function,
                 penalty_constant):
        AugmentedLagrangianOptimizationTarget.__init__(self,
                                                       fairness_rate,
                                                       utility_function,
                                                       fairness_function,
                                                       penalty_constant)
        ManualGradientLagrangianOptimizationTarget.__init__(self,
                                                            fairness_rate,
                                                            utility_function,
                                                            utility_gradient_function,
                                                            fairness_function,
                                                            fairness_gradient_function)

    def model_parameter_gradient(self, **optimization_target_args):
        # lagrangian gradient
        gradient = super().model_parameter_gradient(**optimization_target_args)

        # augmentation
        fairness = self.fairness_function(**optimization_target_args)
        fairness_gradient = self.fairness_function.gradient(**optimization_target_args)
        grad_augmented = self.penalty_constant * fairness * fairness_gradient
        gradient += grad_augmented

        return gradient
