import os
import sys
root_path = os.path.abspath(os.path.join('.'))
if root_path not in sys.path:
    sys.path.append(root_path)

import numpy as np
from copy import deepcopy

from src.policy import LogisticPolicy
from src.util import stack


class BaseLearningAlgorithm():
    def __init__(self, learn_on_entire_history):
        self.learn_on_entire_history = learn_on_entire_history
        self.data_history = None

    def _filter_by_policy(self, data, policy):
        """ Makes decisions for the data based on the specified policy and only returns the x, s and y of the 
        accepted data points, emulating imperfect data collection.
            
        Args:
            data: The dataset of x, s and y based on which the decisions are made.
            policy: The policy used to make the data collection decisions.

        Returns:
            x: The features of the accepted samples
            s: The sensitive attribute of the accepted samples
            y: The ground truth lable of the accepted samples
        """
        x, s, y = data

        decisions = policy(x, s)
        pos_decision_idx = np.expand_dims(np.arange(decisions.shape[0]), axis=1)
        pos_decision_idx = pos_decision_idx[decisions == 1]

        return x[pos_decision_idx], s[pos_decision_idx], y[pos_decision_idx]

    def _minibatch_over_epochs(self, data, batch_size, epochs=1):
        """ Creates minibatches for stochastic gradient ascent according to the epochs and
        batch size.
        
        Args:
            policy: The policy from which the data has been drawn.
            data: The training data
            batch_size: The minibatch size of SGD.
            epochs: The number of epochs the training algorithm will run.
        """     
        for _ in range(0, epochs):
            # only train if there is a large enough sample size to build at least one full batch
            if data["x"].shape[0] < batch_size:
                break

            # minibatching     
            indices = np.random.permutation(data["x"].shape[0]) 
            for batch_start in range(0, len(indices), batch_size):
                batch_end = min(batch_start + batch_size, len(indices))

                x_batch = data["x"][batch_start:batch_end]
                s_batch = data["s"][batch_start:batch_end]
                y_batch = data["y"][batch_start:batch_end]
                ips_weight_batch = data["ips_weights"][batch_start:batch_end]

                yield x_batch, s_batch, y_batch, ips_weight_batch

    def _update_buffer(self, x, s, y, ips_weights):
        """ Update the internal buffer of the training algorithm.
        
        Args:
            policy: The policy from which the data has been drawn.
            x: The features of the n samples
            s: The sensitive attribute of the n samples
            y: The ground truth labels of the n samples
        """
        if (self.learn_on_entire_history and self.data_history is None) or not self.learn_on_entire_history:
            self.data_history = {
                "x": x,
                "s": s,
                "y": y,
                "ips_weights": ips_weights
            }
        elif self.learn_on_entire_history:
            self.data_history["ips_weights"] = np.vstack((self.data_history["ips_weights"], ips_weights))
            self.data_history["x"] = np.vstack((self.data_history["x"], x))
            self.data_history["y"] = np.vstack((self.data_history["y"], y))
            self.data_history["s"] = np.vstack((self.data_history["s"], s))


class ConsequentialLearning(BaseLearningAlgorithm):    
    def __init__(self, learn_on_entire_history):
        """ Creates a new instance of a consequential learning algorithm.
        
        Args:
            learn_on_entire_history: The 
        """
        super(ConsequentialLearning, self).__init__(learn_on_entire_history)

    def _train_model_parameters(self, policy, training_parameters):
        x_test, s_test, _ = training_parameters["data"]["test"]

        # Store initial policy decisions
        decisions_over_time = policy(x_test, s_test).reshape(-1, 1)
        trained_model_parameters = None

        theta_learning_rate = training_parameters["parameter_optimization"]["learning_rate"]
        theta_decay_rate = training_parameters["parameter_optimization"]["decay_rate"]
        theta_decay_step = training_parameters["parameter_optimization"]["decay_step"]
        
        for i in range(0, training_parameters["parameter_optimization"]["time_steps"]): 
            # decay theta learning rate
            if i % theta_decay_step == 0 and i != 0:
                theta_learning_rate *= theta_decay_rate

            # Collect training data
            data = training_parameters["data"]["training"]["theta"][i]   
            x_train, s_train, y_train = self._filter_by_policy(data, policy)
            ips_weights = policy._ips_weights(x_train, s_train)
            self._update_buffer(x_train, s_train, y_train, ips_weights)

            for x, s, y, ips_weights in self._minibatch_over_epochs(
                data=self.data_history, 
                epochs=training_parameters["parameter_optimization"]["epochs"], 
                batch_size=training_parameters["parameter_optimization"]["batch_size"]):
                policy.update_model_parameters(x, s, y, theta_learning_rate, ips_weights)

            # Store decisions made in the time step ...
            decisions = policy(x_test, s_test).reshape(-1, 1)
            decisions_over_time = stack(decisions_over_time, decisions, axis=1)
            
            # ... and the parameters of the model
            trained_model_parameters = policy.get_model_parameters()
        
        return decisions_over_time, trained_model_parameters
        
    def train(self, training_parameters):
        policy = LogisticPolicy(
            training_parameters["model"]["initial_theta"], 
            training_parameters["model"]["fairness_function"], 
            training_parameters["model"]["fairness_gradient_function"], 
            training_parameters["model"]["benefit_function"], 
            training_parameters["model"]["utility_function"], 
            training_parameters["model"]["feature_map"], 
            training_parameters["model"]["initial_lambda"], 
            training_parameters["model"]["use_sensitve_attributes"])

        yield self._train_model_parameters(policy, training_parameters)


class FixedLambdasConsequentialLearning(ConsequentialLearning):    
    def __init__(self, learn_on_entire_history):
        """ Creates a new instance of a dual gradient consequential learning algorithm.
        
        Args:
            learn_on_entire_history: The 
        """
        super(FixedLambdasConsequentialLearning, self).__init__(learn_on_entire_history)
        
    def train(self, training_parameters):
        fairness_rates = training_parameters["model"]["initial_lambda"]

        for fairness_rate in fairness_rates:
            policy = LogisticPolicy(
                training_parameters["model"]["initial_theta"], 
                training_parameters["model"]["fairness_function"], 
                training_parameters["model"]["fairness_gradient_function"], 
                training_parameters["model"]["benefit_function"], 
                training_parameters["model"]["utility_function"], 
                training_parameters["model"]["feature_map"], 
                fairness_rate, 
                training_parameters["model"]["use_sensitve_attributes"])

            yield self._train_model_parameters(policy, training_parameters)


class DualGradientConsequentialLearning(ConsequentialLearning):    
    def __init__(self, learn_on_entire_history):
        """ Creates a new instance of a dual gradient consequential learning algorithm.
        
        Args:
            learn_on_entire_history: The 
        """
        super(DualGradientConsequentialLearning, self).__init__(learn_on_entire_history)
        
    def train(self, training_parameters):
        fairness_rate = training_parameters["model"]["initial_lambda"]
        lambda_learning_rate = training_parameters["lagrangian_optimization"]["learning_rate"]
        lambda_decay_rate = training_parameters["lagrangian_optimization"]["decay_rate"]
        lambda_decay_step = training_parameters["lagrangian_optimization"]["decay_step"]

        policy = LogisticPolicy(
            deepcopy(training_parameters["model"]["initial_theta"]), 
            training_parameters["model"]["fairness_function"], 
            training_parameters["model"]["fairness_gradient_function"], 
            training_parameters["model"]["benefit_function"], 
            training_parameters["model"]["utility_function"], 
            training_parameters["model"]["feature_map"], 
            fairness_rate, 
            training_parameters["model"]["use_sensitve_attributes"])

        for i in range(0, training_parameters["lagrangian_optimization"]["iterations"]):
            # decay lagrangian learning rate
            if i % lambda_decay_step == 0 and i != 0:
                lambda_learning_rate *= lambda_decay_rate
                
            yield self._train_model_parameters(policy, training_parameters)

            # Get lambda training data
            data = training_parameters["data"]["training"]["lambda"]
            x_train, s_train, y_train = self._filter_by_policy(data, policy)

            data = {
                "x": x_train,
                "s": s_train,
                "y": y_train,
                "ips_weights": policy._ips_weights(x_train, s_train)
            }
            # train lambda for the generated training data
            for x, s, y, ips_weights in self._minibatch_over_epochs(
                data=data, 
                epochs=training_parameters["lagrangian_optimization"]["epochs"],
                batch_size=training_parameters["lagrangian_optimization"]["batch_size"]):
                policy.update_fairness_parameter(x, s, y, lambda_learning_rate, ips_weights)

            del x_train, s_train, y_train

                