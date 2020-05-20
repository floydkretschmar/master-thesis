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

    def _clipped_ips_weights(self, policy):
        """ Implements clipped-wIPS as first introduced by https://www.microsoft.com/en-us/research/wp-content/uploads
        /2013/11/bottou13a.pdf to ensure numerical stability of the ips weights for unlikely data samples.
        For data samples with small probability of positive decision by the initial policy pi_0 the IPS weights become
        very large. Clip ips weights to 0 if the probability of a positive decision under the current polity is bigger
        than R * probability of a positive decision under the initial policy. According to the paper R is chosen as the
        fifth largest IPS weight.

        Args:
            policy: The current policy.

        Returns:
            clipped_ips_weights: The clipped ips weights where all weights where prob_current_pi > R * prob_pi_0 are set
            to 0.
        """
        if self.buffer_size > 1:
            sorted_ipsw = np.sort(self.data_history["ips_weights"].squeeze())
            R = sorted_ipsw[-5] if len(sorted_ipsw) >= 5 else sorted_ipsw[0]
        else:
            R = self.data_history["ips_weights"][0]
        _, decision_probabilities = policy(self.data_history["x"], self.data_history["s"])

        clipped_ips_weights = deepcopy(self.data_history["ips_weights"])
        clipped_ips_weights[decision_probabilities >= R * self.data_history["ips_probabilities"]] = 0

        return clipped_ips_weights

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

        decisions, _ = policy(x, s)
        pos_decision_idx = np.expand_dims(np.arange(decisions.shape[0]), axis=1)
        pos_decision_idx = pos_decision_idx[decisions == 1]

        return x[pos_decision_idx], s[pos_decision_idx], y[pos_decision_idx]

    def _update_buffer(self, x, s, y, ips_weights, ips_probabilities):
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
                "ips_weights": ips_weights,
                "ips_probabilities": ips_probabilities
            }
        elif self.learn_on_entire_history:
            self.data_history["ips_weights"] = np.vstack((self.data_history["ips_weights"], ips_weights))
            self.data_history["ips_probabilities"] = np.vstack(
                (self.data_history["ips_probabilities"], ips_probabilities))
            self.data_history["x"] = np.vstack((self.data_history["x"], x))
            self.data_history["y"] = np.vstack((self.data_history["y"], y))
            self.data_history["s"] = np.vstack((self.data_history["s"], s))

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
        num_change_iterations = training_parameters["parameter_optimization"]["num_change_iterations"]
        change_percentage = training_parameters["parameter_optimization"]["change_percentage"]

        decisions, decision_probability = optimizer.policy(self.data_history["x"], self.data_history["s"])
        last_optimization_target = -optimizer.optimization_target(x=self.data_history["x"],
                                                                  s=self.data_history["s"],
                                                                  y=self.data_history["y"],
                                                                  policy=optimizer.policy,
                                                                  decisions=decisions,
                                                                  decision_probabilities=decision_probability,
                                                                  ips_weights=self._clipped_ips_weights(
                                                                      optimizer.policy))
        i = 0
        i_no_change = 0
        # run until convergence or maximum number of epochs is reached
        while epochs < i or i_no_change < num_change_iterations:
            optimizer.update_model_parameters(learning_rate,
                                              batch_size,
                                              self.data_history["x"],
                                              self.data_history["s"],
                                              self.data_history["y"],
                                              self._clipped_ips_weights(optimizer.policy))

            decisions, decision_probability = optimizer.policy(self.data_history["x"], self.data_history["s"])
            current_optimization_target = -optimizer.optimization_target(x=self.data_history["x"],
                                                                         s=self.data_history["s"],
                                                                         y=self.data_history["y"],
                                                                         policy=optimizer.policy,
                                                                         decisions=decisions,
                                                                         decision_probabilities=decision_probability,
                                                                         ips_weights=self._clipped_ips_weights(
                                                                             optimizer.policy))

            # if the change in the last epoch was smaller than the specified percentage of the last optimization target
            # value: increase number of iterations without change by one, otherwise reset
            change = np.abs(last_optimization_target - current_optimization_target)
            if change < np.abs(current_optimization_target * change_percentage):
                i_no_change += 1
            else:
                i_no_change = 0

            last_optimization_target = current_optimization_target
            i += 1

    def train(self, training_parameters):
        """ Executes consequential learning according to the specified training parameters.

        Args:
            training_parameters: The parameters used to configure the consequential learning algorithm.

        Yields:
            statistics: The statistics of the training over all timesteps.
            model_parameters: The model parameters object containing the trained model parameters.
        """
        distribution = training_parameters["distribution"]
        optimization_target = deepcopy(training_parameters["optimization_target"])
        optimizer = deepcopy(training_parameters["model"]).optimizer(
            deepcopy(training_parameters["model"]),
            optimization_target)
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
            data = distribution.sample_train_dataset(
                n_train=training_parameters["data"]["num_train_samples"],
                seed=training_seeds[i])
            x_train, s_train, y_train = self._filter_by_policy(data, optimizer.policy)

            ips_weights, ips_probabilities = optimizer.policy.ips_weights(x_train, s_train)
            self._update_buffer(x_train, s_train, y_train, ips_weights, ips_probabilities)

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
                    for _ in range(0, training_parameters["lagrangian_optimization"]["epochs"]):
                        self._train_model_parameters(optimizer,
                                                     theta_learning_rate,
                                                     training_parameters)

                        optimizer.update_fairness_parameter(lambda_learning_rate,
                                                            training_parameters["lagrangian_optimization"][
                                                                "batch_size"],
                                                            self.data_history["x"],
                                                            self.data_history["s"],
                                                            self.data_history["y"],
                                                            self._clipped_ips_weights(optimizer.policy))
                else:
                    self._train_model_parameters(optimizer,
                                                 theta_learning_rate)

            # Evaluate performance on test set after training ...
            decisions, decision_probabilities = optimizer.policy(x_test, s_test)
            decisions_over_time = stack(decisions_over_time, decisions, axis=1)

            # ... and save the parameters of the model
            parameters = optimizer.parameters
            model_parameters["lambdas"].append(deepcopy(parameters["lambda"]))
            model_parameters["model_parameters"] = [deepcopy(parameters)]

        self._reset_buffer()
        statistics = Statistics.build(
            predictions=decisions_over_time,
            protected_attributes=s_test,
            ground_truths=y_test,
            utility_function=optimization_target.utility_function)

        return statistics, ModelParameters(model_parameters)
