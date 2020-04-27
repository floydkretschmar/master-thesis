import argparse
import os
import sys

module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)

from pathos.pools import _ThreadPool as Pool
from queue import Queue
import multiprocessing as mp
from copy import deepcopy

from src.feature_map import IdentityFeatureMap
from src.functions import cost_utility
from src.plotting import plot_mean, plot_median
from src.training import train
from src.distribution import FICODistribution, COMPASDistribution, AdultCreditDistribution

parser = argparse.ArgumentParser()

parser.add_argument('-d', '--data', type=str, required=True, help="select the distribution (FICO, COMPAS, ADULT)")
parser.add_argument('-c', '--cost', type=float, required=True, help="define the utility cost c")
parser.add_argument('-lr', '--learning_rate', type=float, required=True, help="define the learning rate of theta")
parser.add_argument('-p', '--path', type=str, required=False, help="save path for the result")
parser.add_argument('-ts', '--time_steps', type=int, nargs='+', required=True, help='list of time steps to be used')
parser.add_argument('-e', '--epochs', type=int, nargs='+', required=True, help='list of epochs to be used')
parser.add_argument('-bs', '--batch_sizes', type=int, nargs='+', required=True, help='list of batch sizes to be used')
parser.add_argument('-nb', '--num_batches', type=int, nargs='+', required=True,
                    help='list of number of batches to be used')
parser.add_argument('-i', '--iterations', type=int, required=True, help='the number of internal iterations')
parser.add_argument('-a', '--asynchonous', action='store_true')

args = parser.parse_args()

if args.data == 'FICO':
    distibution = FICODistribution(bias=True, fraction_protected=0.5)
elif args.data == 'COMPAS':
    distibution = COMPASDistribution(bias=True, test_percentage=0.2)
elif args.data == 'ADULT':
    distibution = AdultCreditDistribution(bias=True, test_percentage=0.2)

training_parameters = {
    'distribution': distibution,
    'model': {
        'fairness_function': lambda **fairness_params: [0.0],
        'fairness_gradient_function': lambda **fairness_params: [0.0],
        'utility_function': lambda **util_params: cost_utility(cost_factor=args.cost, **util_params),
        'feature_map': IdentityFeatureMap(distibution.feature_dimension),
        'learn_on_entire_history': False,
        'use_sensitve_attributes': False,
        'bias': True,
        'initial_theta': [0.0, 0.0],
        'initial_lambda': 0.0
    },
    'parameter_optimization': {
        'learning_rate': args.learning_rate,
        'decay_rate': 1,
        'decay_step': 10000,
        'fix_seeds': True
    },
    'test': {
        'num_samples': 10000
    }
}

callback_queue = Queue()
pool = Pool(mp.cpu_count())
thread_count = 0

for time_steps in args.time_steps:
    training_parameters['parameter_optimization']['time_steps'] = time_steps
    for epochs in args.epochs:
        training_parameters['parameter_optimization']['epochs'] = epochs
        for batch_size in args.batch_sizes:
            training_parameters['parameter_optimization']['batch_size'] = batch_size
            for num_batches in args.num_batches:
                training_parameters['parameter_optimization']['num_batches'] = num_batches

                if args.path:
                    training_parameters['save_path'] = args.path
                    training_parameters["save_path"] = "{}/lr{}/ts{}-ep{}-bs{}-nb{}".format(args.path,
                                                                                            args.learning_rate,
                                                                                            time_steps,
                                                                                            epochs,
                                                                                            batch_size,
                                                                                            num_batches)

                pool.apply_async(train,
                                 args=(deepcopy(training_parameters), args.iterations, args.asynchonous),
                                 callback=lambda result: callback_queue.put(result),
                                 error_callback=lambda e: print(e))
                thread_count += 1

best_final_median_utility = float("-inf")
best_utility_path = ""
while thread_count > 0:
    result = callback_queue.get()
    statistics, model_parameters, run_path = result

    final_median_utility = statistics.performance(statistics.UTILITY, statistics.MEDIAN)[-1]

    if final_median_utility > best_final_median_utility:
        best_final_median_utility = final_median_utility
        best_utility_path = run_path

    plot_mean(statistics, "{}/results_mean_time.png".format(run_path))
    plot_median(statistics, "{}/results_median_time.png".format(run_path))
    thread_count -= 1

print(best_utility_path)

pool.close()
pool.join()
