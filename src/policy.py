import sys
import os
import pandas as pd

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import numpy as np
#pylint: disable=no-name-in-module
from scipy.special import expit as sigmoid
from src.util import iterate_minibatches

class BasePolicy():
    """ The base implementation of a policy """

    def __init__(
        self, 
        dim_theta, 
        fairness_function, 
        benefit_value_function, 
        utility_value_function, 
        fairness_rate, 
        use_sensitive_attributes, 
        learn_on_entire_history): 
        """ Initializes a new BasePolicy object.
        
        Args:
            dim_theta: The dimension of the parameters parameterizing the policy model.
            fairness_function: The callback to the function that defines the fairness penalty of the model.
            benefit_value_function: The callback that returns the benefit value for a specified set of data.
            utility_value_function: The callback that returns the utility value for a specified set of data.
            fairness_rate: The fairness rate lambda that regulates the impact of the fairness penalty on the 
            overall utility.
            use_sensitive_attributes: A flag that indicates whether or not the sensitive attributes should be
            used overall or just in evaluating the fairness penalty.
            learn_on_entire_history: A flag that indicates whether or not the policy should be trained on the data
            of all timesteps t < t' or only on the data of the current time step.

        Returns:
            decisions: The decisions.
        """
        self.fairness_function = fairness_function
        self.benefit_value_function = benefit_value_function
        self.utility_value_function = utility_value_function
        self.use_sensitive_attributes = False
        self.fairness_rate = fairness_rate

        self.use_sensitive_attributes = use_sensitive_attributes
        self.learn_on_entire_history = learn_on_entire_history
        self.data_history = None
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

    def _benefit_difference(self, x, s, y, decisions, gradient=False, sampling_data=None):
        """ Calculates the difference of benefits of the given policy for the provided data.
        
        Args:
            x: The features of the n samples
            s: The sensitive attribute of the n samples
            y: The ground truth labels of the n samples
            sampling_data: The dictionary containing the parameters of the distribution that were used to sample 
            the n samples and therefore define the policies by which the samples will be inverse propensity scored. 
            If sampling_data=None no IPS will be applied.

        Returns:
            benefit_difference: The difference in benefits of the policy.
        """
        s_idx = np.arange(s.shape[0]).reshape(-1, 1)
        s_0_idx = s_idx[s == 0]
        s_1_idx = s_idx[s == 1]

        if sampling_data is not None and self.learn_on_entire_history:
            sampling_data_s0 = {
                "theta_idx": sampling_data["theta_idx"][s_0_idx],
                "sampling_thetas": sampling_data["sampling_thetas"]
            }
            sampling_data_s1 = {
                "theta_idx": sampling_data["theta_idx"][s_1_idx],
                "sampling_thetas": sampling_data["sampling_thetas"]
            }
        elif sampling_data is not None:
            sampling_data_s0 = sampling_data
            sampling_data_s1 = sampling_data
        else:
            sampling_data_s0 = None
            sampling_data_s1 = None

        benefit_s0 = self._calculate_expectation(x[s_0_idx], s[s_0_idx], self.benefit_value_function(decisions_s=decisions[s_0_idx], y_s=y[s_0_idx]), gradient, sampling_data_s0)
        benefit_s1 = self._calculate_expectation(x[s_1_idx], s[s_1_idx], self.benefit_value_function(decisions_s=decisions[s_1_idx], y_s=y[s_1_idx]), gradient, sampling_data_s1)

        return benefit_s0 - benefit_s1

    def _calculate_expectation(self, x, s, function_value, gradient=False, sampling_data=None):
        """ Calculates the expectation over the underlying probability distribition of the policy for the 
        specified function value E[target] or the gradient of the policy E[target * \log \grad policy(e | x, s)].
        If specified IPS is applied according to the formula target/pi_theta(e = 1 | x, s) to the function
        value.
        
        Args:
            x: The features of the n samples
            s: The sensitive attribute of the n samples
            function_value: The function value for which the expectation will be calculated.
            gradient: The flag specifying whether the gradient should be calculated instead of the function 
            value expectation.
            sampling_data: The dictionary containing the parameters of the distribution that were used to sample 
            the n samples and therefore define the policies by which the samples will be inverse propensity scored. 
            If sampling_data=None no IPS will be applied.

        Returns:
            expectation or gradient: The expectation over the policy with regard to the specified function 
            value or its gradient.
        """
        raise NotImplementedError("Subclass must override _calculate_expectation(self, x, s, target, gradient=False, sampling_theta=None).")

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

    def _fairness_function(self, x, s, y, decisions, gradient=False, sampling_data=None):
        return self.fairness_function(
            x=x, 
            s=s, 
            y=y, 
            sampling_data=sampling_data, 
            decisions=decisions, 
            gradient=gradient, 
            policy=self)

    def _policy_gradient(self, x, s, y, sampling_data=None):
        """ Calculates the gradient of the policy given the data.
        
        Args:
            x: The features of the n samples
            s: The sensitive attribute of the n samples
            y: The ground truth labels of the n samples
            sampling_data: The dictionary containing the parameters of the distribution that were used to sample 
            the n samples and therefore define the policies by which the samples will be inverse propensity scored. 
            If sampling_data=None no IPS will be applied.

        Returns:
            gradient: The gradient of the policy.
        """
        # make decision according to current policy
        decisions = self(x, s)
        gradient = self._calculate_expectation(x, s, self.utility_value_function(decisions=decisions, y=y), gradient=True, sampling_data=sampling_data) 

        if self.fairness_rate > 0:
            fairness_value = self._fairness_function(x, s, y, decisions, gradient=False, sampling_data=sampling_data)
            fairness_gradient_value = self._fairness_function(x, s, y, decisions, gradient=True, sampling_data=sampling_data)
            grad_fairness = self.fairness_rate * fairness_value * fairness_gradient_value
            gradient += grad_fairness

        return gradient

    def _probability(self, features):
        """ Calculates the probability of a positiv decision given the specified features.
        
        Args:
            features: The features for which the probability will be calculated

        Returns:
            probability: The Probability of a positive decision.
        """
        raise NotImplementedError("Subclass must override calculate probability(features).")

    def benefit_delta(self, x, s, y, decisions):
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
        return np.absolute(self._benefit_difference(x, s, y, decisions))

    def copy(self):
        """ Creates a deep copy of the policy.
        
        Returns:
            copy: The created deep copy.
        """
        raise NotImplementedError("Subclass must override copy(self).")

    def regularized_utility(self, x, s, y, decisions):
        """ Calculates the overall utility of the policy regularized by the fairness constraint.
        
        Args:
            x: The features of the n samples
            s: The sensitive attribute of the n samples
            y: The ground truth labels of the n samples

        Returns:
            utility: The utility of the policy.
        """
        regularized_utility = self._calculate_expectation(x, s, self.utility_value_function(decisions=decisions, y=y))

        if self.fairness_rate > 0:
            fairness_value = self._fairness_function(x, s, y, decisions)
            fairness_penalty = (self.fairness_rate/2) * (fairness_value**2)
            regularized_utility -= fairness_penalty

        return regularized_utility

    def update(self, x, s, y, learning_rate, batch_size, epochs):
        """ Updates the policy parameters using stochastic gradient descent.
        
        Args:
            x: The features of the n samples
            s: The sensitive attribute of the n samples
            y: The ground truth labels of the n samples
            learning_rate: The rate with which the parameters will be updated.
            batch_size: The minibatch size of SGD.

        """
        # if the user specified to keep the history then not just store the data
        # but also the associated sampling thetas for importance sampling
        if self.learn_on_entire_history and self.data_history is None:
            self.data_history = {
                "x": x,
                "s": s,
                "y": y,
                "theta_idx": np.zeros((x.shape[0], 1), dtype=int),
                "sampling_thetas": [self.theta.copy()]
            }
        elif self.learn_on_entire_history:
            theta_idx = self.data_history["theta_idx"].max() + 1
            x = np.vstack((self.data_history["x"], x))
            y = np.vstack((self.data_history["y"], y))
            s = np.vstack((self.data_history["s"], s))
            self.data_history["x"] = x
            self.data_history["y"] = y
            self.data_history["s"] = s
            self.data_history["theta_idx"] = np.vstack((self.data_history["theta_idx"], np.full((x.shape[0], 1), theta_idx, dtype=int)))
            self.data_history["sampling_thetas"].append(self.theta.copy())
        else:
            sampling_data = self.theta.copy()    

        indices = np.arange(x.shape[0])

        for _ in range(0, epochs):
            # minibatching
            np.random.shuffle(indices)
            batch_indices = indices[0:batch_size]

            X_batch = x[batch_indices]
            S_batch = s[batch_indices]
            Y_batch = y[batch_indices]

            if self.learn_on_entire_history:
                sampling_data = {
                    "theta_idx": self.data_history["theta_idx"][batch_indices],
                    "sampling_thetas": self.data_history["sampling_thetas"]
                }

            # calculate the gradient
            gradient = self._policy_gradient(X_batch, S_batch, Y_batch, sampling_data)            

            # update the parameters
            self.theta += learning_rate * gradient

    def utility(self, x, s, y, decisions):
        """ Calculates the utility value or the utility gradient according to the utility vaue function callback specified
        in the constructor.
        
        Args:
            x: The features of the n samples
            s: The sensitive attribute of the n samples
            y: The ground truth labels of the n samples

        Returns:
            utility: The utility value or gradient of the policy.
        """
        return self._calculate_expectation(x, s, self.utility_value_function(decisions=decisions, y=y))

class LogisticPolicy(BasePolicy):
    """ The implementation of the logistic policy. """

    def __init__(self, dim_theta, fairness_function, benefit_value_function, utility_value_function, feature_map, fairness_rate, use_sensitive_attributes, learn_on_entire_history): 
        super(LogisticPolicy, self).__init__(
            dim_theta,
            fairness_function,
            benefit_value_function,
            utility_value_function,
            fairness_rate,
            use_sensitive_attributes,
            learn_on_entire_history)

        self.feature_map = feature_map    

    def copy(self):
        approx_policy = LogisticPolicy(
            self.theta.shape[0],
            self.fairness_function, 
            self.benefit_value_function, 
            self.utility_value_function, 
            self.feature_map, 
            self.fairness_rate, 
            self.use_sensitive_attributes,
            self.learn_on_entire_history) 
        approx_policy.theta = self.theta.copy()
        return approx_policy

    def _probability(self, features):
        return sigmoid(np.matmul(self.feature_map(features), self.theta))

    def _calculate_expectation(self, x, s, function_value, gradient=False, sampling_data=None):
        phi = self.feature_map(self._extract_features(x, s))
        ones = np.ones((function_value.shape[0], 1))

        if gradient:
            distance = np.matmul(phi, self.theta.reshape(-1, 1))
            function_value /= (ones + np.exp(distance))
            function_value = phi * function_value

        if sampling_data is not None and self.learn_on_entire_history:
            theta_idx = sampling_data["theta_idx"]
            sampling_thetas = sampling_data["sampling_thetas"]

            tmp = np.hstack((phi, theta_idx))

            def weighting(row):
                distance = np.matmul(row[:-1], sampling_thetas[int(row[-1])])
                return np.exp(-distance)

            exp = np.apply_along_axis(weighting, 1, tmp).reshape(-1, 1)
            function_value *= (ones + exp)
        elif sampling_data is not None:
            sampling_theta = sampling_data.reshape(-1, 1)
            distance = np.matmul(phi, sampling_theta)
            function_value *= (ones + np.exp(-distance))


        return function_value.mean(axis=0)