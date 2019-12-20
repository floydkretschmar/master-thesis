import sys
import os

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import numpy as np
#pylint: disable=no-name-in-module
from scipy.special import expit as sigmoid
from src.util import iterate_minibatches

class BasePolicy():
    """ The base implementation of a policy """

    def __init__(self, dim_theta, fairness_function, benefit_value_function, utility_value_function, fairness_rate, use_sensitive_attributes): 
        self.fairness_function = fairness_function
        self.benefit_value_function = benefit_value_function
        self.utility_value_function = utility_value_function
        self.use_sensitive_attributes = False
        self.fairness_rate = fairness_rate

        self.use_sensitive_attributes = use_sensitive_attributes
        self.theta = np.zeros(dim_theta)

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

        return np.random.binomial(1, probability).astype(float)

    def _benefit_difference(self, x, s, y, decisions, gradient=False, sampling_theta=None):
        """ Calculates the difference of benefits of the given policy for the provided data.
        
        Args:
            x: The features of the n samples
            s: The sensitive attribute of the n samples
            y: The ground truth labels of the n samples
            sampling_theta: The parameters of the distribution that was used to sample the n samples and therefore defines
            the policy by which the samples will be inverse propensity scored. If sample_theta=None no IPS is applied.

        Returns:
            benefit_difference: The difference in benefits of the policy.
        """
        s_idx = np.arange(s.shape[0]).reshape(-1, 1)
        s_0_idx = s_idx[s == 0]
        s_1_idx = s_idx[s == 1]

        benefit_s0 = self._calculate_expectation(x[s_0_idx], s[s_0_idx], self.benefit_value_function(decisions_s=decisions[s_0_idx], y_s=y[s_0_idx]), gradient, sampling_theta)
        benefit_s1 = self._calculate_expectation(x[s_1_idx], s[s_1_idx], self.benefit_value_function(decisions_s=decisions[s_1_idx], y_s=y[s_1_idx]), gradient, sampling_theta)

        return benefit_s0 - benefit_s1

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

    def _fairness_function(self, x, s, y, decisions, gradient=False, sampling_theta=None):
        return self.fairness_function(x=x, s=s, y=y, decisions=decisions, gradient=gradient, sampling_theta=sampling_theta, policy=self)

    def _probability(self, features):
        """ Calculates the probability of a positiv decision given the specified features.
        
        Args:
            features: The features for which the probability will be calculated

        Returns:
            probability: The Probability of a positive decision.
        """
        raise NotImplementedError("Subclass must override calculate probability(features).")

    def _calculate_expectation(self, x, s, target, gradient=False, sampling_theta=None):
        """ Further processes the result of the utility, benefit and fairness value functions by applying inverse propensity scoring, 
        calculating the gradient or both. Applies IPS according to the formula E[target/pi_theta(e = 1 | x, s)] and/or calculates the 
        gradient of the specified function value according to the formula E[target * \log \grad policy(e | x, s)].
        
        Args:
            x: The features of the n samples
            s: The sensitive attribute of the n samples
            target: The function result for which the gradient will be calculated.
            gradient: The flag specifying whether the gradient should be calculated instead of the function value.
            sample_theta: The parameters of the distribution that was used to sample the n samples and therefore defines
            the policy by which the samples will be inverse propensity scored. If sampling_theta=None IPS is not applied.

        Returns:
            gradient: The gradient of the target.
        """
        raise NotImplementedError("Subclass must override _calculate_expectation(self, x, s, target, gradient=False, sampling_theta=None).")

    def benefit_delta(self, x, s, y, sampling_theta=None):
        """ Calculates the absolute difference of benefits of the given policy for the provided data.
        
        Args:
            x: The features of the n samples
            s: The sensitive attribute of the n samples
            y: The ground truth labels of the n samples
            sampling_theta: The parameters of the distribution that was used to sample the n samples and therefore defines
            the policy by which the samples will be inverse propensity scored. If sample_theta=None no IPS is applied.

        Returns:
            benefit_difference: The difference in benefits of the policy.
        """
        decisions = self(x, s)        
        return np.absolute(self._benefit_difference(x, s, y, decisions, sampling_theta))

    def copy(self):
        """ Creates a deep copy of the policy.
        
        Returns:
            copy: The created deep copy.
        """
        raise NotImplementedError("Subclass must override copy(self).")

    def policy_gradient(self, x, s, y, sampling_theta=None):
        """ Calculates the gradient of the policy given the data.
        
        Args:
            x: The features of the n samples
            s: The sensitive attribute of the n samples
            y: The ground truth labels of the n samples
            sampling_theta: The parameters of the distribution that was used to sample the n samples and therefore defines
            the policy by which the samples will be inverse propensity scored. If sample_theta=None no IPS is applied.

        Returns:
            gradient: The gradient of the policy.
        """
        # make decision according to current policy
        decisions = self(x, s)
        gradient = self._calculate_expectation(x, s, self.utility_value_function(decisions=decisions, y=y), gradient=True, sampling_theta=sampling_theta) 

        if self.fairness_rate > 0:
            fairness_value = self._fairness_function(x, s, y, decisions, gradient=False, sampling_theta=sampling_theta)
            fairness_gradient_value = self._fairness_function(x, s, y, decisions, gradient=True, sampling_theta=sampling_theta)
            grad_fairness = self.fairness_rate * fairness_value * fairness_gradient_value
            gradient += grad_fairness

        return gradient

    def regularized_utility(self, x, s, y, sampling_theta=None):
        """ Calculates the overall utility of the policy regularized by the fairness constraint.
        
        Args:
            x: The features of the n samples
            s: The sensitive attribute of the n samples
            y: The ground truth labels of the n samples
            sampling_theta: The parameters of the distribution that was used to sample the n samples and therefore defines
            the policy by which the samples will be inverse propensity scored. If sample_theta=None no IPS is applied.

        Returns:
            utility: The utility of the policy.
        """
        decisions = self(x, s)
        regularized_utility = self._calculate_expectation(x, s, self.utility_value_function(decisions=decisions, y=y), sampling_theta=sampling_theta)

        if self.fairness_rate > 0:
            fairness_value = self._fairness_function(x, s, y, decisions, sampling_theta=sampling_theta)
            fairness_penalty = (self.fairness_rate/2) * (fairness_value**2)
            regularized_utility -= fairness_penalty

        return regularized_utility

    def update(self, x, s, y, learning_rate, batch_size):
        """ Updates the policy parameters using stochastic gradient descent.
        
        Args:
            x: The features of the n samples
            s: The sensitive attribute of the n samples
            y: The ground truth labels of the n samples
            learning_rate: The rate with which the parameters will be updated.
            batch_size: The minibatch size of SGD.

        """
        sampling_theta = self.theta.copy()    
        #print("Sampling Theta {}".format(sampling_theta))

        for X_batch, S_batch, Y_batch in iterate_minibatches(x, s, y, batch_size):  
            # calculate the gradient
            gradient = self.policy_gradient(X_batch, S_batch, Y_batch, sampling_theta)            

            # update the parameters
            self.theta += learning_rate * gradient 

    def utility(self, x, s, y, sampling_theta=None):
        """ Calculates the utility value or the utility gradient according to the utility vaue function callback specified
        in the constructor.
        
        Args:
            x: The features of the n samples
            s: The sensitive attribute of the n samples
            y: The ground truth labels of the n samples
            sampling_theta: The parameters of the distribution that was used to sample the n samples and therefore defines
            the policy by which the samples will be inverse propensity scored. If sample_theta=None no IPS is applied.

        Returns:
            utility: The utility value or gradient of the policy.
        """
        decisions = self(x, s)
        return self._calculate_expectation(x, s, self.utility_value_function(decisions=decisions, y=y), sampling_theta=sampling_theta)

class LogisticPolicy(BasePolicy):
    """ The implementation of the logistic policy. """

    def __init__(self, dim_theta, fairness_function, benefit_value_function, utility_value_function, feature_map, fairness_rate, use_sensitive_attribute): 
        super(LogisticPolicy, self).__init__(
            dim_theta,
            fairness_function,
            benefit_value_function,
            utility_value_function,
            fairness_rate,
            use_sensitive_attribute)

        self.feature_map = feature_map    

    def copy(self):
        approx_policy = LogisticPolicy(
            self.theta.shape[0],
            self.fairness_function, 
            self.benefit_value_function, 
            self.utility_value_function, 
            self.feature_map, 
            self.fairness_rate, 
            self.use_sensitive_attributes) 
        approx_policy.theta = self.theta.copy()
        return approx_policy

    def _probability(self, features):
        return sigmoid(np.matmul(self.feature_map(features), self.theta))

    def _calculate_expectation(self, x, s, target, gradient=False, sampling_theta=None):
        phi = self.feature_map(self._extract_features(x, s))
        ones = np.ones((target.shape[0], 1))

        if sampling_theta is not None:
            sampling_theta = sampling_theta.reshape(-1, 1)
            target *= ones + np.exp(-np.matmul(phi, sampling_theta))

        if gradient:
            target = (target * phi)/(ones + np.exp(np.matmul(phi, self.theta.reshape(-1, 1))))
            
        return target.mean(axis=0)