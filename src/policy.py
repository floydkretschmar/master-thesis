import os
import sys
root_path = os.path.abspath(os.path.join('.'))
if root_path not in sys.path:
    sys.path.append(root_path)

import numpy as np
#pylint: disable=no-name-in-module
from src.util import sigmoid, check_for_missing_kwargs          

class BasePolicy():
    """ The base implementation of a policy """

    def __init__(
        self, 
        fairness_function, 
        benefit_function, 
        utility_function, 
        fairness_rate, 
        use_sensitive_attributes): 
        """ Initializes a new BasePolicy object.
        
        Args:
            fairness_function: The callback to the function that defines the fairness penalty of the model.
            benefit_function: The callback that returns the benefit value for a specified set of data.
            utility_function: The callback that returns the utility value for a specified set of data.
            fairness_rate: The fairness rate lambda that regulates the impact of the fairness penalty on the 
            overall utility.
            use_sensitive_attributes: A flag that indicates whether or not the sensitive attributes should be
            used overall or just in evaluating the fairness penalty.
        """
        self.fairness_function = fairness_function
        self.benefit_function = benefit_function
        self.utility_function = utility_function
        self.use_sensitive_attributes = False
        self.fairness_rate = fairness_rate

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

        return np.expand_dims(np.random.binomial(1, probability).astype(float), axis=1)

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

    def _fairness_parameter_gradient(self, x, s, y, ips_weights=None):
        """ Calculates the gradient of the the lagrangian multiplier lambda of a policy given the data.
        
        Args:
            x: The features of the n samples
            s: The sensitive attribute of the n samples
            y: The ground truth labels of the n samples
            decisions: The decisions made by the policy based on the features.
            ips_weights: The weights used for inverse propensity scoring. If sampling_data=None 
            no IPS will be applied.

        Returns:
            lambda_gradient: The gradient of the lagrangian multiplier.
        """
        # make decision according to current policy
        decisions = self(x, s)

        fairness = self.fairness_function(
                x=x, 
                s=s, 
                y=y, 
                ips_weights=ips_weights, 
                decisions=decisions, 
                policy=self)
                
        return -(fairness**2)/2

    def _ips_weights(self, x, s):
        """ Calculates the inverse propensity scoring weights according to the formula 1/pi_0(e = 1 | x, s).
        
        Args:
            x: The features of the n samples
            s: The sensitive attribute of the n samples

        Returns:
            ips_weights: The weights for inverse propensity scoring.
        """
        raise NotImplementedError("Subclass must override _ips_weights(self, x, s, sampling_distribution).")

    def _mean_difference(self, target, s):
        """ Calculates the mean difference of the target with regards to the sensitive attribute.
        
        Args:
            target: The target for which the mean difference will be calculated.
            s: The sensitive attribute of the n samples

        Returns:
            mean_difference: The mean difference of the target
        """
        s_idx = np.expand_dims(np.arange(s.shape[0]), axis=1)
        s_0_idx = s_idx[s == 0]
        s_1_idx = s_idx[s == 1]

        if len(s_0_idx) == 0 or len(s_1_idx) == 0:
            return 0.0

        target_s0 = target[s_0_idx].sum(axis=0) / len(s_0_idx)
        target_s1 = target[s_1_idx].sum(axis=0) / len(s_1_idx)

        return target_s0 - target_s1

    def _probability(self, features):
        """ Calculates the probability of a positiv decision given the specified features.
        
        Args:
            features: The features for which the probability will be calculated

        Returns:
            probability: The Probability of a positive decision.
        """
        raise NotImplementedError("Subclass must override calculate probability(features).")    

    def _utility(self, y, decisions, ips_weights=None):
        """ Calculates the utility of the current policy according to the specified utility_function.
        
        Args:
            y: The ground truth labels of the n samples
            decisions: The decisions made by the policy based on the features.
            ips_weights: The weights used for inverse propensity scoring. If sampling_data=None 
            no IPS will be applied.

        Returns:
            utility: The utility of the current policy.
        """
        utility = self.utility_function(decisions=decisions, y=y)

        if ips_weights is not None:
            utility *= ips_weights 

        return utility

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

    def optimization_target(self, x, s, y, decisions, ips_weights=None):
        """ Returns the overall optimization target u(pi) + 1/lambda * F(pi)^2.

        Args:
            x: The features of the n samples
            s: The sensitive attribute of the n samples
            y: The ground truth labels of the n samples
            decisions: The decisions made by the policy based on the features.
            ips_weights: The weights used for inverse propensity scoring. If sampling_data=None 
            no IPS will be applied.
        
        Returns:
            optimization_target: The overall optimization target.
        """
        utility = self._utility(y, decisions, ips_weights)
        fairness = self.fairness_function(
            x=x, 
            s=s, 
            y=y, 
            ips_weights=ips_weights, 
            decisions=decisions, 
            policy=self)

        return np.mean(utility, axis=0) + (self.fairness_rate/2) * (fairness**2)

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
        gradient = self._fairness_parameter_gradient(x, s, y, ips_weights) 
        self.fairness_rate -= learning_rate * gradient         

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
        raise NotImplementedError("Subclass must override update_theta(self, x, s, y, ips_weights).")

class ManualModelParameterGradientPolicy(BasePolicy):
    def __init__(self, fairness_function, fairness_gradient_function, benefit_function, utility_function, fairness_rate, use_sensitive_attributes): 
        super(ManualModelParameterGradientPolicy, self).__init__(
            fairness_function,
            benefit_function,
            utility_function,
            fairness_rate,
            use_sensitive_attributes)

        self.fairness_gradient_function = fairness_gradient_function

    def _log_policy_gradient(self, x, s):
        """ Calculates the gradient of the logarithm of the current policy used to calculate the utility
        and fairness gradients.
        
        Args:
            x: The features of the n samples
            s: The sensitive attribute of the n samples
            sampling_distribution: The distribution pi_0 under which the data has been collected.

        Returns:
            log_gradient: The gradient of the logarithm of the current policy.
        """
        raise NotImplementedError("Subclass must override _log_gradient(self, x, s).")

    def _model_parameter_gradient(self, x, s, y, ips_weights=None):
        """ Calculates the gradient of the the parameters theta of a policy given the data.
        
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

        # calculate utility part of the gradient
        gradient = self._utility_gradient(x, s, y, decisions, ips_weights)

        if self.fairness_rate > 0:
            parameters = {
                "x":x, 
                "s":s, 
                "y":y, 
                "ips_weights":ips_weights, 
                "decisions":decisions, 
                "policy":self
            }
            fairness = self.fairness_function(**parameters)
            fairness_gradient = self.fairness_gradient_function(**parameters)

            grad_fairness = self.fairness_rate * fairness * fairness_gradient
            gradient -= grad_fairness

        return gradient

    def _utility_gradient(self, x, s, y, decisions, ips_weights=None):
        """ Calculates the part of the theta gradient that is defined by the utility function.
        
        Args:
            x: The features of the n samples
            s: The sensitive attribute of the n samples
            y: The ground truth labels of the n samples
            decisions: The decisions made by the policy based on the features.
            ips_weights: The weights used for inverse propensity scoring. If sampling_data=None 
            no IPS will be applied.

        Returns:
            utility_gradient: The partial gradient of the policy.
        """
        log_policy_gradient = self._log_policy_gradient(x, s)
        utility = self._utility(y, decisions, ips_weights)

        utility_grad = log_policy_gradient * utility
        return np.mean(utility_grad, axis=0)

class LogisticPolicy(ManualModelParameterGradientPolicy):
    """ The implementation of the logistic policy. """

    def __init__(self, theta, fairness_function, fairness_gradient_function, benefit_function, utility_function, feature_map, fairness_rate, use_sensitive_attributes): 
        super(LogisticPolicy, self).__init__(
            fairness_function,
            fairness_gradient_function,
            benefit_function,
            utility_function,
            fairness_rate,
            use_sensitive_attributes)

        self.feature_map = feature_map    
        self.theta = np.array(theta)

    def copy(self):
        approx_policy = LogisticPolicy(
            self.theta.copy(),
            self.fairness_function, 
            self.fairness_gradient_function, 
            self.benefit_function, 
            self.utility_function, 
            self.feature_map, 
            self.fairness_rate, 
            self.use_sensitive_attributes) 
        return approx_policy

    def get_model_parameters(self):
        return {
            "theta": self.theta.tolist(), 
            "lambda": self.fairness_rate
        }    

    def update_model_parameters(self, x, s, y, learning_rate, ips_weights=None):
        gradient = self._model_parameter_gradient(x, s, y, ips_weights) 
        self.theta += learning_rate * gradient      

    def _ips_weights(self, x, s):
        phi = self.feature_map(self._extract_features(x, s))

        sampling_theta = np.expand_dims(self.theta, axis=1)
        weights = 1.0 + np.exp(-np.matmul(phi, sampling_theta))

        return weights
        
    def _log_policy_gradient(self, x, s):
        phi = self.feature_map(self._extract_features(x, s))
        #return phi * np.expand_dims(sigmoid(-np.matmul(phi, self.theta)), axis=1)
        return phi / np.expand_dims(1.0 + np.exp(np.matmul(phi, self.theta)), axis=1)  

    def _probability(self, features):
        return sigmoid(np.matmul(self.feature_map(features), self.theta))