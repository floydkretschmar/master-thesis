import os
import sys

import numpy as np
import torch

module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)

from src.functions import cost_utility_probability, cost_utility
from src.plotting import plot_median
from src.training import train
from src.training_evaluation import UTILITY
from src.policy import LogisticPolicy, NeuralNetworkPolicy
from src.distribution import COMPASDistribution
from src.optimization import PenaltyOptimizationTarget

bias = True
distribution = COMPASDistribution(bias=bias, test_percentage=0.2)
dim_theta = distribution.feature_dimension

optim_target = PenaltyOptimizationTarget(0.0,
                                         lambda **util_params: torch.mean(cost_utility_probability(cost_factor=0.5, **util_params)),
                                         lambda **util_params: 0.0)

training_parameters = {
    'model': NeuralNetworkPolicy(distribution.feature_dimension, False),
    'distribution': distribution,
    'optimization_target': optim_target,
    'parameter_optimization': {
        'time_steps': 200,
        'epochs': 50,
        'batch_size': 128,
        'learning_rate': 0.1,
        'learn_on_entire_history': False,
        'fix_seeds': True,
        'clip_weights': True,
        'change_percentage': 0.01
    },
    'data': {
        'num_train_samples': 4096,
        'num_test_samples': 1024
    },
    'evaluation': {
        UTILITY: {
            'measure_function': lambda s, y, decisions: np.mean(cost_utility(y=y, decisions=decisions, cost_factor=0.5)),
            'detailed': False
        }
    }
}

training_parameters["save_path"] = "../res/local_experiments/NO_FAIRNESS"
statistics, model_parameters, run_path = train(
    training_parameters,
    iterations=1,
    asynchronous=False,
    fairness_rates=[0.1])

plot_median(x_values=range(training_parameters["parameter_optimization"]["time_steps"] + 1),
            x_label="Time steps",
            x_scale="linear",
            performance_measures=[statistics.get_additonal_measure(UTILITY, "Utility"),
                                  statistics.demographic_parity(),
                                  statistics.equality_of_opportunity()],
            fairness_measures=[],
            file_path="{}/results_median_time.png".format(run_path))
