import argparse
import os
import sys

module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)

from src.feature_map import IdentityFeatureMap
from src.functions import cost_utility
from src.plotting import plot_mean, plot_median
from src.training import train
from src.distribution import FICODistribution, COMPASDistribution, AdultCreditDistribution

parser = argparse.ArgumentParser()

parser.add_argument('-data', type=str, required=True, help="select the distribution (FICO, COMPAS, ADULT)")
parser.add_argument('-c', type=float, required=True, help="define the utility cost c")
parser.add_argument('-lr', type=float, required=True, help="define the learning rate of theta")
parser.add_argument('-ts', type=int, required=True, help="define the number of timesteps")
parser.add_argument('-e', type=int, required=True, help="define the number of epochs per timestep")
parser.add_argument('-bs', type=int, required=True, help="define the batch size")
parser.add_argument('-nb', type=int, required=True, help="define the number of batches")
args = parser.parse_args()

if args.data == "FICO":
    distibution = FICODistribution(bias=True, fraction_protected=0.5)
elif args.data == "COMPAS":
    distibution = COMPASDistribution(bias=True, test_percentage=0.2)
elif args.data == "ADULT":
    distibution = AdultCreditDistribution(bias=True, test_percentage=0.2)

training_parameters = {
    'save_path': './',
    'model': {
        'fairness_function': lambda **fairness_params: [0.0],
        'fairness_gradient_function': lambda **fairness_params: [0.0],
        'feature_map': IdentityFeatureMap(distibution.feature_dimension),
        'learn_on_entire_history': False,
        'use_sensitve_attributes': False,
        'bias': True,
        'initial_theta': [0.0, 0.0],
        'initial_lambda': 0.0
    },
    'parameter_optimization': {
        'learning_rate': 1,
        'decay_rate': 1,
        'decay_step': 10000,
    },
    'data': {
        'distribution': distibution,
        'num_test_samples': 8192
    }
}

apply_results = []
results_per_iterations = []


def process_result(result):
    statistics, model_parameters, run_path = result
    plot_mean(statistics, "{}/results_mean_time.png".format(run_path))
    plot_median(statistics, "{}/results_median_time.png".format(run_path))


# pool = Pool(mp.cpu_count())

training_parameters['model']['utility_function'] = lambda **util_params: cost_utility(cost_factor=args.c, **util_params)
# for learning_rate in [0.001, 0.01, 0.1, 1]:
training_parameters['parameter_optimization']['learning_rate'] = args.lr
# for time_steps in [50, 100, 200]:
training_parameters['parameter_optimization']['time_steps'] = args.ts
# for epochs in [1, 5, 10]:
training_parameters['parameter_optimization']['epochs'] = args.e
# for batch_size in [50, 250, 500]:
training_parameters['parameter_optimization']['batch_size'] = args.bs
# for num_batches in [10, 100, 1000]:
training_parameters['parameter_optimization']['num_decisions'] = args.bs * args.nb

training_parameters["experiment_name"] = "FICO/lr{}/ts{}-ep{}-bs{}-nb{}".format(args.lr, args.ts, args.e, args.bs,
                                                                                args.nb)

# pool.apply_async(train, args=(training_parameters, 15, True), callback=process_result, error_callback=lambda e: print(e))
process_result(train(training_parameters, iterations=30, asynchronous=True))

# pool.close()
# pool.join()
