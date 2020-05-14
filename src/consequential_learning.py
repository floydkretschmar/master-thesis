import os
import sys
from copy import deepcopy

import numpy as np

root_path = os.path.abspath(os.path.join('.'))
if root_path not in sys.path:
    sys.path.append(root_path)

from src.util import stack, get_random, train_test_split

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

    def _minibatch(self, data, batch_size):
        """ Creates minibatches for stochastic gradient ascent according to the epochs and
        batch size.
        
        Args:
            policy: The policy from which the data has been drawn.
            data: The training data
            batch_size: The minibatch size of SGD.
            epochs: The number of epochs the training algorithm will run.
        """
        # minibatching
        indices = get_random().permutation(data["x"].shape[0])
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

    def _train_model_parameters(self, policy, optimizer, training_parameters):
        """ Executes the training algorithm.

        Args:
            training_parameters: The parameters used to configure the consequential learning algorithm.

        Yields:
            decisions_over_time: The decisions made by a policy at timestep t over all time steps
            trained_model_parameters: The model parameters at the final timestep T
        """

    def train(self, training_parameters):
        """ Executes consequential learning.

        Args:
            training_parameters: The parameters used to configure the consequential learning algorithm.

        Yields:
            decisions_over_time: The decisions made by a policy at timestep t over all time steps for a single lambda.
            trained_model_parameters: The model parameters at the final timestep T
        """
        distribution = training_parameters["distribution"]
        policy = training_parameters["model"]
        optimization_target = training_parameters["optimization_target"]
        x_test, s_test, y_test = training_parameters["test"]

        optimizer = policy.optimizer(optimization_target)

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

        for i in range(0, training_parameters["parameter_optimization"]["time_steps"]):
            # decay theta learning rate
            if i % theta_decay_step == 0 and i != 0:
                theta_learning_rate *= theta_decay_rate

            # Collect training data
            x, y, s = distribution.sample_train_dataset(
                n_train=int(((training_parameters["parameter_optimization"]["batch_size"]
                              * training_parameters["parameter_optimization"]["num_batches"]) / 90) * 100),
                seed=training_parameters["parameter_optimization"]["seeds"][i]
                if "seeds" in training_parameters["parameter_optimization"] else None)
            x, x_validate, y, y_validate, s, s_validate = train_test_split(x, y, s, test_size=0.1)
            x_train, s_train, y_train = self._filter_by_policy((x, s, y), policy)

            ips_weights = policy.ips_weights(x_train, s_train)
            self._update_buffer(x_train, s_train, y_train, ips_weights)

            last_optimization_target = 0
            iterations_with_deterioration = 0
            deterioration_iterations = training_parameters["parameter_optimization"][
                "deterioration_iterations"] if "deterioration_iterations" in training_parameters[
                "parameter_optimization"] else None

            # train if at least one full batch can be formed from filtered data
            for epoch in range(0, training_parameters["parameter_optimization"]["epochs"]):
                ##### TRAIN THETA #####
                for x, s, y, ips_weights_batch in self._minibatch(
                        data=self.data_history,
                        batch_size=training_parameters["parameter_optimization"]["batch_size"]):
                    optimizer.update_model_parameters(x, s, y, theta_learning_rate, ips_weights_batch)

                if deterioration_iterations:
                    d, dp = policy(x_validate, s_validate)
                    current_optimization_target = -optimization_target(policy, x_validate, s_validate, y_validate, d,
                                                                       dp)
                    change = current_optimization_target - last_optimization_target

                    if change > 0:
                        iterations_with_deterioration = 0
                    else:
                        iterations_with_deterioration += 1

                    if iterations_with_deterioration > deterioration_iterations:
                        break
                    else:
                        last_optimization_target = current_optimization_target

            ##### TRAIN LAMBDA #####
            if "lagrangian_optimization" in training_parameters:
                lambda_learning_rate = training_parameters["lagrangian_optimization"]["learning_rate"]
                for _ in range(0, training_parameters["lagrangian_optimization"]["epochs"]):
                    # train lambda for the generated training data
                    for x, s, y, ips_weights_batch in self._minibatch(
                            data=self.data_history,
                            batch_size=training_parameters["lagrangian_optimization"]["batch_size"]):
                        optimizer.update_fairness_parameter(x,
                                                            s,
                                                            y,
                                                            lambda_learning_rate,
                                                            ips_weights_batch)

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
        return (decisions_over_time,
                np.array(fairness_over_time, dtype=float).reshape(-1, 1),
                np.array(utilities_over_time, dtype=float).reshape(-1, 1)), \
               model_parameters
