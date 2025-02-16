import os
import sys
from copy import deepcopy

import numpy as np
import torch

root_path = os.path.abspath(os.path.join('.'))
if root_path not in sys.path:
    sys.path.append(root_path)

from src.util import stack, train_test_split, stable_divide, abs, sign
from src.training_evaluation import Statistics, ModelParameters
from src.plotting import plot_epoch_statistics
from src.optimization import PytorchStochasticGradientOptimizer


class BaseLearningAlgorithm:
    def __init__(self, learn_on_entire_history):
        """ Creates a new instance of a learning algorithm.

        Args:
            learn_on_entire_history: The flag indicating whether the algorithm should learn on the entire history or
            just the last time step.
        """
        self.learn_on_entire_history = learn_on_entire_history
        self.data_history = None

    @property
    def buffer_size(self):
        """ Returns the number of individuals currently stored in the data buffer. """
        if self.data_history is not None:
            return len(self.data_history["s"])
        else:
            return 0

    def _filter_by_policy(self, data, policy):
        """ Makes decisions for the data based on the specified policy pi_0 and only returns the x, s and y of the
        accepted individuals, emulating imperfect data collection according to some initial decision policy pi_0.

        Args:
            data: The dataset of x, s and y based on which the decisions are made.
            policy: The policy used to make the data collection decisions.

        Returns:
            x: The features of the accepted samples
            s: The sensitive attribute of the accepted samples
            y: The ground truth lable of the accepted samples
        """
        x, s, y = data

        decisions, probabilities = policy(x, s)
        pos_decision_idx = np.arange(decisions.shape[0]).reshape(-1, 1)
        pos_decision_idx = pos_decision_idx[decisions == 1]

        return x[pos_decision_idx], s[pos_decision_idx], y[pos_decision_idx], probabilities[pos_decision_idx]

    def _update_buffer(self, x, s, y, ips_weights):
        """ Update the internal buffer of the training algorithm. If learn_on_entire_history is specified, new data will
        be appended, otherwise new data will replace existing data.

        Args:
            x: The features of the n samples
            s: The sensitive attribute of the n samples
            y: The ground truth labels of the n samples
            ips_weights: The ips weights of the n samples according to the initial policy pi_0.
            ips_probabilities: The probabilities of a decision for the n samples according to the initial policy pi_0.
        """
        if (self.learn_on_entire_history and self.data_history is None) or not self.learn_on_entire_history:
            self.data_history = {
                "x": x,
                "s": s,
                "y": y,
                "ips_weights": ips_weights
            }
        elif self.learn_on_entire_history:
            self.data_history["ips_weights"] = stack(self.data_history["ips_weights"], ips_weights, 0)
            self.data_history["x"] = stack(self.data_history["x"], x, 0)
            self.data_history["y"] = stack(self.data_history["y"], y, 0)
            self.data_history["s"] = stack(self.data_history["s"], s, 0)

    def _reset_buffer(self):
        """ Resets the interal data storage. """
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

    def _train_model_parameters(self, optimizer, learning_rate, training_parameters):
        """ Executes the core training of the model parameters theta.

        Args:
            optimizer: The optimization algorithm used to train the policy.
            learning_rate: The learning rate with which the policy is updated.
            batch_size: The size of the batches into which the training data will be split.
            epochs: The maximum amount of epochs for which the model training runs. Default is None, meaning the
            training runs until convergence.

        Yields:
            decisions_over_time: The decisions made by a policy at timestep t over all time steps
            trained_model_parameters: The model parameters at the final timestep T
        """
        batch_size = training_parameters["parameter_optimization"]["batch_size"]
        epochs = training_parameters["parameter_optimization"]["epochs"]
        num_change_iterations = training_parameters["parameter_optimization"]["change_iterations"]
        change_percentage = training_parameters["parameter_optimization"]["change_percentage"]

        x_train, x_val, s_train, s_val, y_train, y_val, ip_weights_train, ip_weights_val = train_test_split(
            self.data_history["x"],
            self.data_history["s"],
            self.data_history["y"],
            self.data_history["ips_weights"],
            test_percentage=0.2)

        if len(x_train) > 0 and len(x_val) > 0:
            with torch.no_grad():
                decisions, decision_probability = optimizer.policy(x_val, s_val)

            last_optimization_target = -optimizer.optimization_target(x=x_val,
                                                                      s=s_val,
                                                                      y=y_val,
                                                                      policy=optimizer.policy,
                                                                      decisions=decisions,
                                                                      decision_probabilities=decision_probability,
                                                                      ips_weights=ip_weights_val)
            i = 0
            i_no_change = 0
            # run until convergence or maximum number of epochs is reached
            while i_no_change < num_change_iterations:
                optimizer.update_model_parameters(learning_rate,
                                                  batch_size,
                                                  x_train,
                                                  s_train,
                                                  y_train,
                                                  ip_weights_train)

                with torch.no_grad():
                    decisions, decision_probability = optimizer.policy(x_val, s_val)

                current_optimization_target = -optimizer.optimization_target(x=x_val,
                                                                             s=s_val,
                                                                             y=y_val,
                                                                             policy=optimizer.policy,
                                                                             decisions=decisions,
                                                                             decision_probabilities=decision_probability,
                                                                             ips_weights=ip_weights_val)

                # if the change in the last epoch was smaller than the specified percentage of the last optimization target
                # value: increase number of iterations without change by one, otherwise reset
                # s = sign(last_optimization_target)
                # if current_optimization_target <= (last_optimization_target * (1 + change_percentage * s)):
                new_target = last_optimization_target + abs((last_optimization_target * change_percentage))
                if current_optimization_target <= new_target:
                    i_no_change += 1
                else:
                    i_no_change = 0

                last_optimization_target = current_optimization_target
                i += 1
                if epochs <= i:
                    break

    def train(self, training_parameters):
        """ Executes consequential learning according to the specified training parameters.

        Args:
            training_parameters: The parameters used to configure the consequential learning algorithm.

        Yields:
            statistics: The statistics of the training over all timesteps.
            model_parameters: The model parameters object containing the trained model parameters.
        """
        distribution = training_parameters["distribution"]
        policy = deepcopy(training_parameters["model"])
        optim_target = training_parameters["optimization_target"]
        dual_optimization = "lagrangian_optimization" in training_parameters
        if dual_optimization:
            optimizer_args = {
                "fairness_training_algorithm": training_parameters["lagrangian_optimization"]["training_algorithm"],
                "policy_training_algorithm": training_parameters["parameter_optimization"]["training_algorithm"]
            }
        else:
            optimizer_args = {
                "policy_training_algorithm": training_parameters["parameter_optimization"]["training_algorithm"]
            }

        optimizer = policy.optimizer(optim_target, **optimizer_args)
        as_tensor = isinstance(optimizer, PytorchStochasticGradientOptimizer)

        # Get test data
        x_test, s_test, y_test = distribution.sample_test_dataset(
            n_test=training_parameters["data"]["num_test_samples"], as_tensor=as_tensor)

        # Store initial policy decisions
        with torch.no_grad():
            decisions_over_time, decision_probabilities = optimizer.policy(x_test, s_test)

        model_parameters = {
            "lambdas": [optimizer.parameters["lambda"]],
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
            data = distribution.sample_train_dataset(n_train=training_parameters["data"]["num_train_samples"],
                                                     as_tensor=as_tensor)
            x_train, s_train, y_train, pi_0_probabilities = self._filter_by_policy(data, optimizer.policy)

            if x_train.shape[0] > 0:
                with torch.no_grad():
                    ips_weights = stable_divide(1, pi_0_probabilities)

                    if training_parameters["parameter_optimization"]["clip_weights"]:
                        s_1_prob = s_train.sum() / len(s_train)
                        s_0_prob = (1 - s_train).sum() / len(s_train)
                        s_1_idx = np.where(s_train == 1)
                        s_0_idx = np.where(s_train == 0)
                        ips_weights[s_1_idx] = ips_weights[s_1_idx] * s_1_prob
                        ips_weights[s_0_idx] = ips_weights[s_0_idx] * s_0_prob

                self._update_buffer(x_train, s_train, y_train, ips_weights)
            elif x_train.shape[0] == 0 and not self.learn_on_entire_history:
                self.data_history = None

            # only train if there is actual training data
            if self.buffer_size > 0:
                if dual_optimization:
                    # decay lambda learning rate
                    if i % lambda_decay_step == 0 and i != 0:
                        lambda_learning_rate *= lambda_decay_rate

                    # for the dual gradient algorithm:
                    # 1. Train model parameters (until convergence)
                    # 2. Update fairness rate
                    # 3. Repeat 1. for #epochs
                    optimizer.create_policy_checkpoint()
                    for j in range(0, training_parameters["lagrangian_optimization"]["epochs"]):
                        optimizer.restore_last_policy_checkpoint()
                        self._train_model_parameters(optimizer,
                                                     theta_learning_rate,
                                                     training_parameters)

                        gradient = optimizer.update_fairness_parameter(lambda_learning_rate,
                                                                       training_parameters["lagrangian_optimization"][
                                                                           "batch_size"],
                                                                       self.data_history["x"],
                                                                       self.data_history["s"],
                                                                       self.data_history["y"],
                                                                       self.data_history["ips_weights"])
                else:
                    self._train_model_parameters(optimizer,
                                                 theta_learning_rate,
                                                 training_parameters)
            # Evaluate performance on test set after training ...
            with torch.no_grad():
                decisions, decision_probabilities = optimizer.policy(x_test, s_test)
            decisions_over_time = stack(decisions_over_time, decisions, axis=1)

            fairness = optimizer.optimization_target.fairness_function(s=s_test,
                                                                       decisions=decisions,
                                                                       decision_probabilities=decisions,
                                                                       y=y_test)
            util = optimizer.optimization_target.utility_function(decisions=decisions,
                                                                  decision_probabilities=decisions,
                                                                  y=y_test)
            print("Timestep {}: Lambda {} \t Fairness (test) {} \t Utility (test) {}".
                  format(i, optimizer.optimization_target.fairness_rate, fairness, util))

            # ... and save the parameters of the model
            parameters = optimizer.parameters
            model_parameters["lambdas"].append(deepcopy(parameters["lambda"]))
            model_parameters["model_parameters"] = [deepcopy(parameters)]

        self._reset_buffer()
        return_data = []
        for data in [decisions_over_time, s_test, y_test]:
            if torch.is_tensor(data):
                return_data.append(data.numpy())
            else:
                return_data.append(data)

        decisions_over_time, s_test, y_test = tuple(return_data)
        statistics = Statistics(
            predictions=decisions_over_time,
            protected_attributes=s_test,
            ground_truths=y_test,
            additonal_measures=training_parameters["evaluation"] if "evaluation" in training_parameters else None)

        return statistics, ModelParameters(model_parameters)
