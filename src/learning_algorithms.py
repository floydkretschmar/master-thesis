import os
import sys
root_path = os.path.abspath(os.path.join('.'))
if root_path not in sys.path:
    sys.path.append(root_path)
    
import numpy as np
from copy import deepcopy

from src.util import check_for_missing_kwargs  

class BaseLearningAlgorithm():
    def __init__(self, learn_on_entire_history):
        self.learn_on_entire_history = learn_on_entire_history
        self.data_history = None

    def update(self, sample_policy, x, s, y, **optimization_args):
        """ Preprocesses and updates the policy parameters using a strategy defined by the class.
        
        Args:
            sample_policy: The policy from which the data has been drawn.
            x: The features of the n samples
            s: The sensitive attribute of the n samples
            y: The ground truth labels of the n samples
            update_args: The additional parameters used by specific training algorithms
        """
        ips_weights = sample_policy._ips_weights(x, s, sample_policy)

        if self.learn_on_entire_history and self.data_history is None:
            self.data_history = {
                "x": x,
                "s": s,
                "y": y,
                "ips_weights": ips_weights
            }
        elif self.learn_on_entire_history:
            x = np.vstack((self.data_history["x"], x))
            y = np.vstack((self.data_history["y"], y))
            s = np.vstack((self.data_history["s"], s))
            ips_weights = np.vstack((self.data_history["ips_weights"], ips_weights))
            self.data_history["ips_weights"] = ips_weights
            self.data_history["x"] = x
            self.data_history["y"] = y
            self.data_history["s"] = s

        update_policy = deepcopy(sample_policy)
        self._update_core(update_policy, x, s, y, ips_weights, **optimization_args)
        return update_policy

    def _update_core(self, current_policy, x, s, y, ips_weights, **optimization_args):
        raise NotImplementedError("Subclass must override _ips_weights(self, x, s, sampling_distribution).")


class StochasticGradientAscent(BaseLearningAlgorithm):
    def __init__(self, learn_on_entire_history):
        super(StochasticGradientAscent, self).__init__(learn_on_entire_history)

    def _minibatch(self, x, s, y, ips_weights, epochs, batch_size):
        """ Creates minibatches for stochastic gradient ascent according to the epochs and
        batch size.
        
        Args:
            sample_policy: The policy from which the data has been drawn.
            x: The features of the n samples
            s: The sensitive attribute of the n samples
            y: The ground truth labels of the n samples
            ips_weights: The weights used for inverse propensity scoring. If sampling_data=None 
            epochs: The number of epochs the training algorithm will run.
            batch_size: The minibatch size of SGD.
        """
        for _ in range(0, epochs):
            # only train if there is a large enough sample size to build at least one full batch
            if x.shape[0] < batch_size:
                break

            # minibatching     
            indices = np.random.permutation(x.shape[0]) 
            for batch_start in range(0, len(indices), batch_size):
                batch_end = min(batch_start + batch_size, len(indices))

                X_batch = x[batch_start:batch_end]
                S_batch = s[batch_start:batch_end]
                Y_batch = y[batch_start:batch_end]
                ips_weights_batch = ips_weights[batch_start:batch_end]

                yield X_batch, S_batch, Y_batch, ips_weights_batch

    def _update_parameters(self, current_policy, x, s, y, ips_weights, learning_rates):
        """ Updates the policy parameters using stochastic gradient ascent.
        
        Args:
            sample_policy: The policy from which the data has been drawn.
            x: The features of the n samples
            s: The sensitive attribute of the n samples
            y: The ground truth labels of the n samples
            learning_rates: The dictionary of learning rates used to update the parameters.
                theta: The learning rate for the model parameters theta. If the learning
                rate for theta does not exist, then theta is fixed and will not be updated.
                lambda: The learning rate for the lagrangien multiplier lambda. If the learning
                rate for lambda does not exist, then lambda is fixed and will not be updated.
        """
        if "theta" in learning_rates:
            # calculate the gradient for theta
            gradient = current_policy._theta_gradient(x, s, y, ips_weights)   
            # update theta
            current_policy.theta += learning_rates["theta"] * gradient

        if "lambda" in learning_rates:
            # calculate the gradient for lamba
            gradient = current_policy._lambda_gradient(x, s, y, ips_weights)   
            # update lambda
            current_policy.fairness_rate -= learning_rates["lambda"] * gradient

    def _update_core(self, current_policy, x, s, y, ips_weights, **optimization_args):
        """ Updates the policy parameters using to stochastic gradient ascent.
        
        Args:
            sample_policy: The policy from which the data has been drawn.
            x: The features of the n samples
            s: The sensitive attribute of the n samples
            y: The ground truth labels of the n samples
            ips_weights: The weights used for inverse propensity scoring. If sampling_data=None 
            optimization_args: The additional parameters for STG include:
                epochs: The number of epochs the training algorithm will run.
                batch_size: The minibatch size of SGD.
                learning_rates: The dictionary of learning rates used to update the parameters.
                    theta: The learning rate for the model parameters theta. If the learning
                    rate for theta does not exist, then theta is fixed and will not be updated.
                    lambda: The learning rate for the lagrangien multiplier lambda. If the learning
                    rate for lambda does not exist, then lambda is fixed and will not be updated.
        """
        check_for_missing_kwargs("STGTraining.update()", ["epochs", "learning_rates", "batch_size"], optimization_args)

        for X_batch, S_batch, Y_batch, ips_weights_batch in self._minibatch(x, s, y, ips_weights, optimization_args["epochs"], optimization_args["batch_size"]):
            self._update_parameters(current_policy, X_batch, S_batch, Y_batch, ips_weights_batch, optimization_args["learning_rates"])
