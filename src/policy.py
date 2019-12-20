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

    def __init__(self, fairness_function, benefit_value_function, utility_value_function, fairness_rate, dim_x, dim_s=None): 
        self.fairness_function = fairness_function
        self.benefit_value_function = benefit_value_function
        self.utility_value_function = utility_value_function
        self.use_s = False
        self.fairness_rate = fairness_rate

        dim_theta = dim_x
        self.dim_x = dim_x
        self.dim_s = None
        if dim_s is not None:
            self.use_s = True
            dim_theta += dim_s
            self.dim_s = dim_s

        self.theta = np.zeros(dim_theta)

    def __call__(self, features):
        """ Returns the decisions made by the policy for x and s.
        
        Args:
            x: The features of the n samples
            s: The sensitive attribute of the n samples

        Returns:
            decisions: The decisions.
        """
        probability = self.probability(features)

        return np.random.binomial(1, probability).astype(float)
    
    def benefit(self, features_s, decisions_s, y_s, gradient=False, sampling_theta=None):
        """ Calculates the benefit value or the benefit gradient according to the benefit value function callback specified
        in the constructor.
        
        Args:
            x_s: The features of the n_s samples where s is the same.
            s: The sensitive attributes of the n_s samples
            y_s: The ground truth labels of the n_s samples where s is the same.

        Returns:
            benefit: The benefit value or gradient of the policy.
        """
        raise NotImplementedError("Subclass must override calculate benefit_function(self, x_s, s, sample_theta, gradient).")

    def benefit_delta(self, features, decisions, s, y):
        features_tuple, decisions_tuple, y_tuple = self.extract_fairness_tuples(features, decisions, s, y)
        return np.absolute(self.benefit_difference(features_tuple, decisions_tuple, y_tuple))

    def benefit_difference(self, features_tuple, decisions_tuple, y_tuple, gradient=False, sampling_theta=None):
        """ Calculates the difference of benefits or the difference of benefit gradients of the given policy.
        
        Args:
            x: The features of the n samples
            s: The sensitive attribute of the n samples
            y: The ground truth labels of the n samples
            gradient: The flag specifying whether the gradient should be calculated instead of the function value.
            sample_theta: The parameters of the distribution that was used to sample the n samples and therefore defines
            the policy by which the samples will be inverse propensity scored. If sample_theta=None no IPS is applied.

        Returns:
            benefit_difference: The difference in benefits of the policy.
        """
        return self.benefit(features_tuple[0], decisions_tuple[0], y_tuple[0], gradient, sampling_theta) - self.benefit(features_tuple[1], decisions_tuple[1], y_tuple[1], gradient, sampling_theta)

    def copy(self):
        """ Creates a deep copy of the policy.
        
        Returns:
            copy: The created deep copy.
        """
        raise NotImplementedError("Subclass must override copy(self).")

    def extract_features(self, x, s):
        """ Extracts the relevant features from the sample.
        
        Args:
            x: The features of the n samples
            s: The sensitive attribute of the n samples

        Returns:
            features: The relevant features of the samples.
        """
        if self.use_s:
            features = np.concatenate((x, s), axis=1)
        else:
            features = x

        return features

    def extract_fairness_tuples(self, features, decisions, s, y):
        s_idx = np.arange(s.shape[0]).reshape(-1, 1)
        s_0_idx = s_idx[s == 0]
        s_1_idx = s_idx[s == 1]
        decisions_tuple = [decisions[s_0_idx], decisions[s_1_idx]]
        y_tuple = [y[s_0_idx], y[s_1_idx]]
        features_tuple = [features[s_0_idx], features[s_1_idx]]

        return features_tuple, decisions_tuple, y_tuple

    def fairness(self, features_tuple, decisions_tuple, y_tuple, gradient=False, sampling_theta=None):
        """ Calculates the fairness value or the fairness gradient according to the fairness value function callback specified
        in the constructor.
        
        Args:
            x: The features of the n samples \n
            s: The sensitive attribute of the n samples
            y: The ground truth labels of the n samples
            gradient: The flag specifying whether the gradient should be calculated instead of the function value.
            sample_theta: The parameters of the distribution that was used to sample the n samples and therefore defines
            the policy by which the samples will be inverse propensity scored. If sample_theta=None no IPS is applied.

        Returns:
            fairness: The fairness value or gradient of the policy.
        """
        return self.fairness_function(features_tuple=features_tuple, decisions_tuple=decisions_tuple, y_tuple=y_tuple, gradient=gradient, sampling_theta=sampling_theta, policy=self)

    def policy_gradient(self, x, s, y, sampling_theta=None):
        """ Calculates the overall gradient of the policy.
        
        Args:
            x: The features of the n samples
            s: The sensitive attribute of the n samples
            y: The ground truth labels of the n samples
            sample_theta: The parameters of the distribution that was used to sample the n samples and therefore defines
            the policy by which the samples will be inverse propensity scored. If sample_theta=None no IPS is applied.


        Returns:
            gradient: The gradient of the policy.
        """
        # make decision according to current policy
        features = self.extract_features(x, s)
        decisions = self(features)

        gradient = self.utility(features, decisions, y, gradient=True, sampling_theta=sampling_theta)

        if self.fairness_rate > 0:
            features_tuple, decisions_tuple, y_tuple = self.extract_fairness_tuples(features, decisions, s, y)

            grad_fairness_penalty = self.fairness(features_tuple, decisions_tuple, y_tuple, gradient=False, sampling_theta=sampling_theta) * self.fairness(features_tuple, decisions_tuple, y_tuple, gradient=True, sampling_theta=sampling_theta)
            gradient += self.fairness_rate * grad_fairness_penalty

        return gradient

    def probability(self, features):
        """ Calculates the probability of a positiv decision given the specified features.
        
        Args:
            features: The features for which the probability will be calculated

        Returns:
            probability: The Probability of a positive decision.
        """
        raise NotImplementedError("Subclass must override calculate probability(features).")

    def process_function_result(self, features, target, gradient=False, sampling_theta=None):
        """ Further processes the result of the utility, benefit and fairness value functions by applying inverse propensity scoring, 
        calculating the gradient or both. Applies IPS according to the formula E[target/pi_theta(e = 1 | x, s)] and/or calculates the 
        gradient of the specified function value according to the formula E[target * \log \grad policy(e | x, s)].
        
        Args:
            features: The features of the n samples
            target: The function result for which the gradient will be calculated.
            gradient: The flag specifying whether the gradient should be calculated instead of the function value.
            sample_theta: The parameters of the distribution that was used to sample the n samples and therefore defines
            the policy by which the samples will be inverse propensity scored. If sampling_theta=None IPS is not applied.

        Returns:
            gradient: The gradient of the target.
        """
        raise NotImplementedError("Subclass must override target_gradient(self, x, s, target, sampling_theta=None).")

    def regularized_utility(self, features, decisions, s, y, sampling_theta=None):
        """ Calculates the overall utility of the policy regularized by the fairness constraint.
        
        Args:
            x: The features of the n samples
            s: The sensitive attribute of the n samples
            y: The ground truth labels of the n samples

        Returns:
            utility: The utility of the policy.
        """
        regularized_utility = self.utility(features, decisions, y, sampling_theta=sampling_theta)

        if self.fairness_rate > 0:
            features_tuple, decisions_tuple, y_tuple = self.extract_fairness_tuples(features, decisions, s, y)   
            fairness_penalty = (self.fairness_rate/2) * (self.fairness(features_tuple, decisions_tuple, y_tuple, sampling_theta=sampling_theta)**2)
            regularized_utility -= fairness_penalty

        return regularized_utility

    def update(self, data, learning_rate, batch_size):
        """ Updates the policy parameters using stochastic gradient descent.
        
        Args:
            x: The features of the n samples
            s: The sensitive attribute of the n samples
            target: The function result for which the gradient will be calculated.
            sample_theta: The parameters of the distribution that was used to sample the n samples and therefore defines
            the policy by which the samples will be inverse propensity scored. If sample_theta=None no IPS is applied.


        Returns:
            gradient: The gradient of the target.
        """
        x, s, y = data
        sampling_theta = self.theta.copy()    
        #print("Sampling Theta {}".format(sampling_theta))

        for X_batch, S_batch, Y_batch in iterate_minibatches(x, s, y, batch_size):  
            # calculate the gradient
            gradient = self.policy_gradient(X_batch, S_batch, Y_batch, sampling_theta)            

            # update the parameters
            self.theta += learning_rate * gradient 
            #print("Updated Theta {}".format(self.theta))

    def utility(self, features, decisions, y, gradient=False, sampling_theta=None):
        """ Calculates the utility value or the utility gradient according to the utility vaue function callback specified
        in the constructor.
        
        Args:
            x: The features of the n samples
            s: The sensitive attribute of the n samples
            y: The ground truth labels of the n samples
            gradient: The flag specifying whether the gradient should be calculated instead of the function value.
            sample_theta: The parameters of the distribution that was used to sample the n samples and therefore defines
            the policy by which the samples will be inverse propensity scored. If sample_theta=None no IPS is applied.

        Returns:
            utility: The utility value or gradient of the policy.
        """
        raise NotImplementedError("Subclass must override utility(self, x, s, gradient=False, sampling_theta=None).")

class LogisticPolicy(BasePolicy):
    """ The implementation of the logistic policy. """

    def __init__(self, fairness_function, benefit_value_function, utility_value_function, feature_map, fairness_rate, dim_x, dim_s=None): 
        super(LogisticPolicy, self).__init__(
            fairness_function,
            benefit_value_function,
            utility_value_function,
            fairness_rate,
            dim_x,
            dim_s)

        self.feature_map = feature_map
    
    def benefit(self, features_s, decisions_s, y_s, gradient=False, sampling_theta=None):
        benefit_params = {
            "features_s": features_s, 
            "decisions_s": decisions_s, 
            "y_s": y_s
        }
        benefit_value = self.benefit_value_function(**benefit_params)        
        return self.process_function_result(features_s, benefit_value, gradient, sampling_theta)

    def copy(self):
        approx_policy = LogisticPolicy(
            self.fairness_function, 
            self.benefit_value_function, 
            self.utility_value_function, 
            self.feature_map, 
            self.fairness_rate, 
            self.dim_x, 
            self.dim_s) 
        approx_policy.theta = self.theta.copy()
        return approx_policy

    def probability(self, features):
        return sigmoid(np.matmul(self.feature_map(features), self.theta))

    def process_function_result(self, features, target, gradient=False, sampling_theta=None):
        phi = self.feature_map(features)
        ones = np.ones((target.shape[0], 1))

        if sampling_theta is not None:
            sampling_theta = sampling_theta.reshape(-1, 1)
            ips_weight = ones + np.exp(-np.matmul(phi, sampling_theta))
            target *= ips_weight

        if gradient:
            log_gradient_denomniator = ones + np.exp(np.matmul(phi, self.theta.reshape(-1, 1)))
            target = (target * phi)/log_gradient_denomniator
            
        return target.mean(axis=0)

    def utility(self, features, decisions, y, gradient=False, sampling_theta=None):
        utility_params = {
            "features": features, 
            "decisions": decisions, 
            "y": y
        }
        utility_value = self.utility_value_function(**utility_params)        
        return self.process_function_result(features, utility_value, gradient, sampling_theta)

