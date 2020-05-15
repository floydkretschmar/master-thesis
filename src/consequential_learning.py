import os
import sys
from copy import deepcopy

import numpy as np

root_path = os.path.abspath(os.path.join('.'))
if root_path not in sys.path:
    sys.path.append(root_path)

from src.util import stack
from src.training_evaluation import Statistics, ModelParameters


class BaseLearningAlgorithm:
    def __init__(self, learn_on_entire_history):
        self.learn_on_entire_history = learn_on_entire_history
        self.data_history = None

    @property
    def buffer_size(self):
        if self.data_history is not None:
            return self.data_history["x"].shape[0]
        else:
            return 0

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

        decisions, _ = policy(x, s)
        pos_decision_idx = np.expand_dims(np.arange(decisions.shape[0]), axis=1)
        pos_decision_idx = pos_decision_idx[decisions == 1]

        return x[pos_decision_idx], s[pos_decision_idx], y[pos_decision_idx]

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

    def _reset_buffer(self):
        self.data_history = None

    def train(self, training_parameters):
        """ Executes the training algorithm.

        Args:
            training_parameters: The parameters used to configure the consequential learning algorithm.

        Yields:
            decisions_over_time: The decisions made by a policy at timestep t over all time steps
            trained_model_parameters: The model parameters at the final timestep T
        """
        raise NotImplementedError("Subclass must override train(self, training_parameters).")


class ConsequentialLearning(BaseLearningAlgorithm):
    def __init__(self, learn_on_entire_history):
        """ Creates a new instance of a consequential learning algorithm.
        
        Args:
            learn_on_entire_history: The flag indicating whether the algorithm should learn on the entire history or
            just the last time step.
        """
        super(ConsequentialLearning, self).__init__(learn_on_entire_history)

    def _train_model_parameters(self, optimizer, epochs, learning_rate, batch_size):
        """ Executes the core training of the model parameters theta.

        Args:
            optimizer: The optimization algorithm used to train the policy.
            epochs: The amount of epochs for which the model training runs.
            learning_rate: The learning rate with which the policy is updated.
            batch_size: The size of the batches into which the training data will be split.

        Yields:
            decisions_over_time: The decisions made by a policy at timestep t over all time steps
            trained_model_parameters: The model parameters at the final timestep T
        """
        for _ in range(0, epochs):
            optimizer.update_model_parameters(learning_rate,
                                              batch_size,
                                              self.data_history["x"],
                                              self.data_history["s"],
                                              self.data_history["y"],
                                              self.data_history["ips_weights"])

    def train(self, training_parameters):
        """ Executes consequential learning.

        Args:
            training_parameters: The parameters used to configure the consequential learning algorithm.

        Yields:
            decisions_over_time: The decisions made by a policy at timestep t over all time steps for a single lambda.
            trained_model_parameters: The model parameters at the final timestep T
        """
        distribution = training_parameters["distribution"]
        policy = deepcopy(training_parameters["model"])
        optimization_target = deepcopy(training_parameters["optimization_target"])
        optimizer = policy.optimizer(optimization_target)
        dual_optimization = "lagrangian_optimization" in training_parameters

        # Prepare data seeds
        training_seeds = training_parameters["data"]["training_seeds"] if "training_seeds" in training_parameters[
            "data"] else [None] * training_parameters["parameter_optimization"]["time_steps"]
        test_seed = training_parameters["data"]["test_seed"] if "test_seed" in training_parameters[
            "data"] else None

        # Get test data
        x_test, s_test, y_test = distribution.sample_test_dataset(
            n_test=training_parameters["data"]["num_test_samples"],
            seed=test_seed)

        # Store initial policy decisions
        decisions_over_time, decision_probabilities = policy(x_test, s_test)
        fairness_over_time = [optimization_target.fairness_function(
            policy=policy,
            x=x_test,
            s=s_test,
            y=y_test,
            decisions=decisions_over_time,
            decision_probabilities=decision_probabilities)]
        utilities_over_time = [optimization_target.utility_function(
            policy=policy,
            x=x_test,
            s=s_test,
            y=y_test,
            decisions=decisions_over_time,
            decision_probabilities=decision_probabilities)]

        model_parameters = {
            "lambdas": [optimizer.get_parameters()["lambda"]],
            "model_parameters": None
        }
        theta_learning_rate = training_parameters["parameter_optimization"]["learning_rate"]
        theta_decay_rate = training_parameters["parameter_optimization"]["decay_rate"]
        theta_decay_step = training_parameters["parameter_optimization"]["decay_step"]

        if dual_optimization:
            lambda_learning_rate = training_parameters["lagrangian_optimization"]["learning_rate"]
            lambda_decay_rate = training_parameters["lagrangian_optimization"]["decay_rate"]
            lambda_decay_step = training_parameters["lagrangian_optimization"]["decay_step"]

        for i in range(0, training_parameters["parameter_optimization"]["time_steps"]):
            # decay theta learning rate
            if i % theta_decay_step == 0 and i != 0:
                theta_learning_rate *= theta_decay_rate

            # Collect training data
            data = distribution.sample_train_dataset(
                n_train=training_parameters["data"]["num_train_samples"],
                seed=training_seeds[i])
            x_train, s_train, y_train = self._filter_by_policy(data, policy)

            ips_weights = policy.ips_weights(x_train, s_train)
            self._update_buffer(x_train, s_train, y_train, ips_weights)

            if dual_optimization:
                # decay lambda learning rate
                if i % lambda_decay_step == 0 and i != 0:
                    lambda_learning_rate *= lambda_decay_rate

                # for the dual gradient algorithm:
                # 1. Train model parameters (until convergence)
                # 2. Update fairness rate
                # 3. Repeat 1. for #epochs
                optimizer.create_checkpoint()
                for _ in range(0, training_parameters["lagrangian_optimization"]["epochs"]):
                    self._train_model_parameters(optimizer,
                                                 training_parameters["parameter_optimization"]["epochs"],
                                                 theta_learning_rate,
                                                 training_parameters["parameter_optimization"]["batch_size"])

                    optimizer.update_fairness_parameter(lambda_learning_rate,
                                                        training_parameters["lagrangian_optimization"]["batch_size"],
                                                        self.data_history["x"],
                                                        self.data_history["s"],
                                                        self.data_history["y"],
                                                        self.data_history["ips_weights"])
                    optimizer.restore_checkpoint()

            self._train_model_parameters(optimizer,
                                         training_parameters["parameter_optimization"]["epochs"],
                                         theta_learning_rate,
                                         training_parameters["parameter_optimization"]["batch_size"])

            # Evaluate performance on test set after training ...
            decisions, decision_probabilities = policy(x_test, s_test)
            decisions_over_time = stack(decisions_over_time, decisions, axis=1)
            fairness_over_time.append(optimization_target.fairness_function(
                policy=policy,
                x=x_test,
                s=s_test,
                y=y_test,
                decisions=decisions,
                decision_probabilities=decision_probabilities))
            utilities_over_time.append(optimization_target.utility_function(
                policy=policy,
                x=x_test,
                s=s_test,
                y=y_test,
                decisions=decisions,
                decision_probabilities=decision_probabilities))

            # ... and save the parameters of the model
            parameters = optimizer.get_parameters()
            model_parameters["lambdas"].append(deepcopy(parameters["lambda"]))
            model_parameters["model_parameters"] = [deepcopy(parameters)]

        self._reset_buffer()
        statistics = Statistics.build(
            predictions=decisions_over_time,
            observations=x_test,
            fairness=np.array(fairness_over_time, dtype=float).reshape(-1, 1),
            utility=np.array(utilities_over_time, dtype=float).reshape(-1, 1),
            protected_attributes=s_test,
            ground_truths=y_test)

        return statistics, ModelParameters(model_parameters)
