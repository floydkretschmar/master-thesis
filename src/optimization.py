import os
import sys
root_path = os.path.abspath(os.path.join('.'))
if root_path not in sys.path:
    sys.path.append(root_path)

import numpy as np

from src.policy import ManualGradientPolicy


class OptimizationTarget():
    def __init__(self, initial_fairness_rate):
        super().__init__()
        self.fairness_rate = initial_fairness_rate

    def __call__(self, utility, fairness):
        raise NotImplementedError("Subclass must override __call__(self, utility, fairness).")

    def gradient_wrt_theta(self, utility, utility_gradient, fairness, fairness_gradient):
        raise NotImplementedError("Subclass must override gradient_wrt_theta(self, utility, utility_gradient, fairness, fairness_gradient).")

    def gradient_wrt_lambda(self, fairness):
        raise NotImplementedError("Subclass must override gradient_wrt_lambda(self, fairness).")

class PenaltyOptimizationTarget(OptimizationTarget):
    def __init__(self, initial_fairness_rate):
        super().__init__(initial_fairness_rate)
        
    def __call__(self, utility, fairness):
        return utility - (self.fairness_rate/2) * fairness**2

    def gradient_wrt_theta(self, utility, utility_gradient, fairness, fairness_gradient):
        gradient = utility_gradient

        if self.fairness_rate > 0:
            grad_fairness = self.fairness_rate * fairness * fairness_gradient
            gradient -= grad_fairness

        return gradient

    def gradient_wrt_lambda(self, fairness):
        return -(fairness**2)/2

class LagrangianOptimizationTarget(OptimizationTarget):
    def __init__(self, initial_fairness_rate):
        super().__init__(initial_fairness_rate)
        
    def __call__(self, utility, fairness):
        return utility - (self.fairness_rate * fairness)

    def gradient_wrt_theta(self, utility, utility_gradient, fairness, fairness_gradient):
        gradient = utility_gradient

        if self.fairness_rate > 0:
            grad_fairness = self.fairness_rate * fairness_gradient
            gradient -= grad_fairness

        return gradient

    def gradient_wrt_lambda(self, fairness):                
        return -fairness


class Optimizer():
    def __init__(self, policy, optimization_target, utility_function, fairness_function):
        super().__init__()
        self.optimization_target = optimization_target
        self.utility_function = utility_function
        self.fairness_function = fairness_function
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
        raise NotImplementedError("Subclass must override gradient_wrt_lambda(self, x, s, y, decisions, ips_weights=None).")

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
        
        # calculate relevant fairness value
        fairness_parameters = {
            "x":x, 
            "s":s, 
            "y":y, 
            "ips_weights":ips_weights, 
            "decisions":decisions, 
            "policy":self.policy
        }
        fairness = self.fairness_function(**fairness_parameters)

        # call the optimization target for gradient calculation
        gradient = self.optimization_target.gradient_wrt_lambda(fairness)
        self.optimization_target.fairness_rate -= learning_rate * gradient    


class ManualGradientOptimizer(Optimizer):
    def __init__(self, manual_gradient_policy, optimization_target, utility_function, fairness_function, fairness_gradient_function):
        super().__init__(manual_gradient_policy, optimization_target, utility_function, fairness_function)
        self.fairness_gradient_function = fairness_gradient_function
        assert isinstance(manual_gradient_policy, ManualGradientPolicy), "{} is not a ManualGradientPolicy. ManualGradientOptimizer can only be used with a ManualGradientPolicy.".format(manual_gradient_policy.__name__)

    def _utility(self, x, s, y, decisions, ips_weights=None, gradient=False):
        """ Calculates the utility or utility gradient of the current policy according to the specified utility_function.
        
        Args:
            x: The features of the n samples
            s: The sensitive attribute of the n samples
            y: The ground truth labels of the n samples
            decisions: The decisions made by the policy based on the features.
            ips_weights: The weights used for inverse propensity scoring. If sampling_data=None 
            no IPS will be applied.

        Returns:
            utility/utility_gradient: The utility (gradient) of the current policy.
        """
        utility = self.utility_function(decisions=decisions, y=y)

        if ips_weights is not None:
            utility *= ips_weights 

        if not gradient:
            return utility.mean(axis=0)
        else:
            log_policy_gradient = self.policy.log_policy_gradient(x, s)
            utility_grad = log_policy_gradient * utility
            return np.mean(utility_grad, axis=0)

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
        # make decision according to current policy
        decisions = self.policy(x, s)

        # calculate relevant utility and fairness values
        utility = self._utility(x, s, y, decisions, ips_weights)
        utility_gradient = self._utility(x, s, y, decisions, ips_weights, gradient=True)

        fairness_parameters = {
            "x":x, 
            "s":s, 
            "y":y, 
            "ips_weights":ips_weights, 
            "decisions":decisions, 
            "policy":self.policy
        }
        fairness = self.fairness_function(**fairness_parameters)
        fairness_gradient = self.fairness_gradient_function(**fairness_parameters)

        # call the optimization target for gradient calculation
        gradient = self.optimization_target.gradient_wrt_theta(utility, utility_gradient, fairness, fairness_gradient)
        self.policy.theta += learning_rate * gradient       