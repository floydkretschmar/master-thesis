import sys
import os
import pandas as pd

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import numpy as np
#pylint: disable=no-name-in-module
from src.util import sigmoid, check_for_missing_kwargs          

class BasePolicy():
    """ The base implementation of a policy """

    def __init__(
        self, 
        theta, 
        fairness_gradient_function, 
        benefit_function, 
        utility_function, 
        fairness_rate, 
        use_sensitive_attributes): 
        """ Initializes a new BasePolicy object.
        
        Args:
            dim_theta: The dimension of the parameters parameterizing the policy model.
            fairness_gradient_function: The callback to the function that defines the fairness penalty of the model.
            benefit_function: The callback that returns the benefit value for a specified set of data.
            utility_function: The callback that returns the utility value for a specified set of data.
            fairness_rate: The fairness rate lambda that regulates the impact of the fairness penalty on the 
            overall utility.
            use_sensitive_attributes: A flag that indicates whether or not the sensitive attributes should be
            used overall or just in evaluating the fairness penalty.
            learn_on_entire_history: A flag that indicates whether or not the policy should be trained on the data
            of all timesteps t < t' or only on the data of the current time step.
        """
        self.fairness_gradient_function = fairness_gradient_function
        self.benefit_function = benefit_function
        self.utility_function = utility_function
        self.use_sensitive_attributes = False
        self.fairness_rate = fairness_rate

        self.use_sensitive_attributes = use_sensitive_attributes
        self.theta = np.array(theta)

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

    def _ips_weights(self, x, s, sampling_distribution):
        """ Calculates the inverse propensity scoring weights according to the formula 1/pi_0(e = 1 | x, s).
        
        Args:
            x: The features of the n samples
            s: The sensitive attribute of the n samples
            sampling_distribution: The distribution pi_0 under which the data has been collected.

        Returns:
            ips_weights: The weights for inverse propensity scoring.
        """
        raise NotImplementedError("Subclass must override _ips_weights(self, x, s, sampling_distribution).")

    def _log_gradient(self, x, s):
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

    def _lambda_gradient(self, x, s, y, decisions, ips_weights=None):
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

        raise NotImplementedError("Subclass must override _lambda_gradient(self, x, s, y, ips_weights=None).")

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

    def _theta_gradient(self, x, s, y, ips_weights=None):
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
            fairness, fairness_gradient = self.fairness_gradient_function(
                x=x, 
                s=s, 
                y=y, 
                ips_weights=ips_weights, 
                decisions=decisions, 
                policy=self)
            grad_fairness = self.fairness_rate * fairness * fairness_gradient
            gradient -= grad_fairness

        return gradient


    def _probability(self, features):
        """ Calculates the probability of a positiv decision given the specified features.
        
        Args:
            features: The features for which the probability will be calculated

        Returns:
            probability: The Probability of a positive decision.
        """
        raise NotImplementedError("Subclass must override calculate probability(features).")

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
        numerator, denominator = self._log_gradient(x, s)
        utility = self.utility_function(decisions=decisions, y=y)
        utility_grad = utility / denominator 

        if ips_weights is not None:
            utility_grad *= ips_weights 

        utility_grad = numerator * utility_grad          
        return np.mean(utility_grad, axis=0)

    def benefit_delta(self, x, s, y, decisions):
        """ Calculates the absolute difference of benefits of the given policy for the provided data.
        
        Args:
            x: The features of the n samples
            s: The sensitive attribute of the n samples
            y: The ground truth labels of the n samples
            decisions: The decisions made by the policy based on the features.

        Returns:
            benefit_delta: The absolute difference in benefits of the policy.
        """    
        return self._mean_difference(self.benefit_function(decisions=decisions, x=x, s=s, y=y), s)

    def copy(self):
        """ Creates a deep copy of the policy.
        
        Returns:
            copy: The created deep copy.
        """
        raise NotImplementedError("Subclass must override copy(self).")

    def utility(self, x, s, y, decisions):
        """ Calculates the utility value or the utility gradient according to the utility vaue function callback specified
        in the constructor.
        
        Args:
            x: The features of the n samples
            s: The sensitive attribute of the n samples
            y: The ground truth labels of the n samples
            decisions: The decisions made by the policy based on the features.

        Returns:
            utility: The utility value or gradient of the policy.
        """
        return self.utility_function(decisions=decisions, x=x, s=s, y=y).mean(axis=0)

class LogisticPolicy(BasePolicy):
    """ The implementation of the logistic policy. """

    def __init__(self, theta, fairness_gradient_function, benefit_function, utility_function, feature_map, fairness_rate, use_sensitive_attributes): 
        super(LogisticPolicy, self).__init__(
            theta,
            fairness_gradient_function,
            benefit_function,
            utility_function,
            fairness_rate,
            use_sensitive_attributes)

        self.feature_map = feature_map    

    def copy(self):
        approx_policy = LogisticPolicy(
            self.theta.shape[0],
            self.fairness_gradient_function, 
            self.benefit_function, 
            self.utility_function, 
            self.feature_map, 
            self.fairness_rate, 
            self.use_sensitive_attributes) 
        approx_policy.theta = self.theta.copy()
        return approx_policy

    def _ips_weights(self, x, s, sampling_distribution):
        phi = sampling_distribution.feature_map(sampling_distribution._extract_features(x, s))

        sampling_theta = np.expand_dims(sampling_distribution.theta.copy(), axis=1)
        weights = 1.0 + np.exp(-np.matmul(phi, sampling_theta))

        return weights

    def _lambda_gradient(self, x, s, y, decisions, ips_weights=None):
        fairness, _ = self.fairness_gradient_function(
                x=x, 
                s=s, 
                y=y, 
                ips_weights=ips_weights, 
                decisions=decisions, 
                policy=self)
        return fairness

    def _log_gradient(self, x, s):
        phi = self.feature_map(self._extract_features(x, s))
        #return phi * np.expand_dims(sigmoid(-np.matmul(phi, self.theta)), axis=1)
        return phi, np.expand_dims(1.0 + np.exp(np.matmul(phi, self.theta)), axis=1)  

    def _probability(self, features):
        return sigmoid(np.matmul(self.feature_map(features), self.theta))
