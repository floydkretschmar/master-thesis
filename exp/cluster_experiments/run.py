import argparse
import os
import sys

import numpy as np

module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.feature_map import IdentityFeatureMap
from src.functions import cost_utility
from src.plotting import plot_mean, plot_median
from src.training import train
from src.training_evaluation import UTILITY, COVARIANCE_OF_DECISION_DP
from src.distribution import FICODistribution, COMPASDistribution, AdultCreditDistribution, GermanCreditDistribution
from src.util import mean_difference, get_list_of_seeds
from src.optimization import PenaltyOptimizationTarget, LagrangianOptimizationTarget, \
    AugmentedLagrangianOptimizationTarget
from src.policy import LogisticPolicy


# region Fairness Definitions

def calc_benefit(decisions, ips_weights):
    if ips_weights is not None:
        decisions *= ips_weights

    return decisions


def calc_covariance(s, decisions, ips_weights):
    new_s = 1 - (2 * s)

    if ips_weights is not None:
        mu_s = np.mean(new_s * ips_weights, axis=0)
        d = decisions * ips_weights
    else:
        mu_s = np.mean(new_s, axis=0)
        d = decisions

    covariance = (new_s - mu_s) * d
    return covariance


def fairness_function_gradient(type, **fairness_kwargs):
    policy = fairness_kwargs["policy"]
    x = fairness_kwargs["x"]
    s = fairness_kwargs["s"]
    y = fairness_kwargs["y"]
    decisions = fairness_kwargs["decisions"]
    ips_weights = fairness_kwargs["ips_weights"]

    if type == "BD_DP" or type == "BD_EOP":
        result = calc_benefit(decisions, ips_weights)
    elif type == "COV_DP":
        result = calc_covariance(s, decisions, ips_weights)

    log_gradient = policy.log_policy_gradient(x, s)
    grad = log_gradient * result

    if type == "BD_DP":
        return mean_difference(grad, s)
    elif type == "COV_DP":
        return np.mean(grad, axis=0)
    elif type == "BD_EOP":
        y1_indices = np.where(y == 1)
        return mean_difference(grad[y1_indices], s[y1_indices])


def fairness_function(type, **fairness_kwargs):
    s = fairness_kwargs["s"]
    decisions = fairness_kwargs["decisions"]
    ips_weights = fairness_kwargs["ips_weights"]
    y = fairness_kwargs["y"]

    if type == "BD_DP":
        benefit = calc_benefit(decisions, ips_weights)
        return mean_difference(benefit, s)
    elif type == "COV_DP":
        covariance = calc_covariance(s, decisions, ips_weights)
        return np.mean(covariance, axis=0)
    elif type == "BD_EOP":
        benefit = calc_benefit(decisions, ips_weights)
        y1_indices = np.where(y == 1)
        return mean_difference(benefit[y1_indices], s[y1_indices])


# endregion

def single_run(args):
    if args.fairness_type:
        fair_fct = lambda **fairness_params: fairness_function(type=args.fairness_type, **fairness_params)
        fair_fct_grad = lambda **fairness_params: fairness_function_gradient(type=args.fairness_type, **fairness_params)
        initial_lambda = args.fairness_value
    else:
        fair_fct = lambda **fairness_params: fairness_function(type='BD_DP', **fairness_params)
        fair_fct_grad = lambda **fairness_params: fairness_function_gradient(type='BD_DP', **fairness_params)
        initial_lambda = 0.0

    if args.data == 'FICO':
        distibution = FICODistribution(bias=True, fraction_protected=0.5)
    elif args.data == 'COMPAS':
        distibution = COMPASDistribution(bias=True, test_percentage=0.2)
    elif args.data == 'ADULT':
        distibution = AdultCreditDistribution(bias=True, test_percentage=0.2)
    elif args.data == 'GERMAN':
        distibution = GermanCreditDistribution(bias=True, test_percentage=0.3)

    training_parameters = {
        'model': {
            'constructor': LogisticPolicy,
            'parameters': {
                "theta": np.zeros((distibution.feature_dimension)),
                "feature_map": IdentityFeatureMap(distibution.feature_dimension),
                "use_sensitive_attributes": False
            }
        },
        'distribution': distibution,
        'optimization_target': {
            'constructor': PenaltyOptimizationTarget,
            'parameters': {
                'fairness_function': fair_fct,
                'fairness_gradient_function': fair_fct_grad,
                'utility_function': lambda **util_params: cost_utility(cost_factor=args.cost, **util_params)
            }
        },
        'parameter_optimization': {
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'learning_rate': args.learning_rate,
            'learn_on_entire_history': False,
            'time_steps': args.time_steps,
            'clip_weights': args.ip_weight_clipping,
            'change_percentage': args.change_percentage,
            'change_iterations': args.change_iterations
        },
        'data': {
            'num_train_samples': args.num_samples,
            'num_test_samples': 10000,
            'fix_seeds': True
        },
        'evaluation': {
            UTILITY: {
                'measure_function': lambda s, y, decisions: np.mean(cost_utility(cost_factor=args.cost,
                                                                                 s=s,
                                                                                 y=y,
                                                                                 decisions=decisions)),
                'detailed': False
            },
            COVARIANCE_OF_DECISION_DP: {
                'measure_function': lambda s, y, decisions: fairness_function(
                    type="COV_DP",
                    x=None,
                    s=s,
                    y=y,
                    decisions=decisions,
                    ips_weights=None,
                    policy=None),
                'detailed': False
            }
        }
    }

    if args.seed_path:
        del training_parameters["data"]["fix_seeds"]

        if os.path.isfile(args.seed_path):
            seeds = np.load(args.seed_path)
            training_parameters['data']["training_seeds"] = seeds["train"]
            training_parameters['data']["test_seed"] = seeds["test"]
        else:
            train_seeds = get_list_of_seeds(200)
            test_seeds = get_list_of_seeds(1)
            training_parameters['data']["training_seeds"] = train_seeds
            training_parameters['data']["test_seed"] = test_seeds
            np.savez(args.seed_path, train=train_seeds, test=test_seeds)

    if args.path:
        if args.fairness_type is not None:
            training_parameters["save_path"] = "{}/c{}/lr{}/ts{}-ep{}-bs{}".format(args.path,
                                                                                   args.cost,
                                                                                   args.learning_rate,
                                                                                   args.time_steps,
                                                                                   args.epochs,
                                                                                   args.batch_size)

            if args.fairness_learning_rate is not None:
                subfolder = "flr{}/-fe{}-fbs{}".format(args.fairness_learning_rate,
                                                       args.fairness_epochs,
                                                       args.fairness_batch_size)
            else:
                subfolder = args.fairness_value

            if args.process_id is not None:
                training_parameters["save_path_subfolder"] = "{}/{}".format(subfolder, args.process_id)
            else:
                training_parameters["save_path_subfolder"] = subfolder
        else:
            training_parameters["save_path"] = "{}/no_fairness/c{}/lr{}/ts{}-ep{}-bs{}".format(args.path,
                                                                                               args.cost,
                                                                                               args.learning_rate,
                                                                                               args.time_steps,
                                                                                               args.epochs,
                                                                                               args.batch_size)
            if args.process_id is not None:
                training_parameters["save_path_subfolder"] = args.process_id

    if args.fairness_type is not None and args.fairness_learning_rate is not None:
        training_parameters["lagrangian_optimization"] = {
            'epochs': args.fairness_epochs,
            'batch_size': args.fairness_batch_size,
            'learning_rate': args.fairness_learning_rate
        }
        if not args.fairness_augmented:
            training_parameters["optimization_target"]["constructor"] = LagrangianOptimizationTarget
        else:
            training_parameters["optimization_target"]["constructor"] = AugmentedLagrangianOptimizationTarget
            training_parameters["optimization_target"]["parameters"]['penalty_constant'] \
                = training_parameters['lagrangian_optimization']['learning_rate']

    statistics, model_parameters, run_path = train(
        training_parameters,
        iterations=args.iterations,
        asynchronous=args.asynchronous,
        fairness_rates=[initial_lambda])

    if args.plot:
        plot_mean(x_values=range(training_parameters["parameter_optimization"]["time_steps"] + 1),
                  x_label="Time steps",
                  x_scale="linear",
                  performance_measures=[statistics.get_additonal_measure(UTILITY, "Utility"),
                                        statistics.accuracy()],
                  fairness_measures=[statistics.demographic_parity(),
                                     statistics.equality_of_opportunity(),
                                     statistics.get_additonal_measure(COVARIANCE_OF_DECISION_DP,
                                                                      "Covariance of Decision (DP)")],
                  file_path="{}/results_mean_time.png".format(run_path))
        plot_median(x_values=range(training_parameters["parameter_optimization"]["time_steps"] + 1),
                    x_label="Time steps",
                    x_scale="linear",
                    performance_measures=[statistics.get_additonal_measure(UTILITY, "Utility"),
                                          statistics.accuracy()],
                    fairness_measures=[statistics.demographic_parity(),
                                       statistics.equality_of_opportunity(),
                                       statistics.get_additonal_measure(COVARIANCE_OF_DECISION_DP,
                                                                        "Covariance of Decision (DP)")],
                    file_path="{}/results_median_time.png".format(run_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-sp', '--seed_path', type=str, required=False, help="path for the seeds .npz file")

    parser.add_argument('-d', '--data', type=str, required=True,
                        help="select the distribution (FICO, COMPAS, ADULT, GERMAN)")
    parser.add_argument('-c', '--cost', type=float, required=True, help="define the utility cost c")
    parser.add_argument('-lr', '--learning_rate', type=float, required=True, help="define the learning rate of theta")
    parser.add_argument('-p', '--path', type=str, required=False, help="save path for the result")
    parser.add_argument('-ts', '--time_steps', type=int, required=True, help='number of time steps to be used')
    parser.add_argument('-e', '--epochs', type=int, required=True, help='number of epochs to be used')
    parser.add_argument('-bs', '--batch_size', type=int, required=True, help='batch size to be used')
    parser.add_argument('-ci', '--change_iterations', type=int, required=False, default=5,
                        help='the number of iterations without the amout of percentage improvemnt specified by '
                             '--change_percentage after which the training of the policy will be stopped.'
                             '(Default = 5)')
    parser.add_argument('-cp', '--change_percentage', type=int, required=False, default=0.05,
                        help='the percentage of improvement per training epoch that is considered the minimum amount of'
                             'improvement. (Default = 0.05)')
    parser.add_argument('-ns', '--num_samples', type=int, required=True, help='number of batches to be used')
    parser.add_argument('-i', '--iterations', type=int, required=True, help='the number of internal iterations')
    parser.add_argument('-ipc', '--ip_weight_clipping', action='store_true')
    parser.add_argument('-a', '--asynchronous', action='store_true')
    parser.add_argument('--plot', required=False, action='store_true')
    parser.add_argument('-pid', '--process_id', type=str, required=False, help="process id for identification")

    parser.add_argument('-f', '--fairness_type', type=str, required=False,
                        help="select the type of fairness (BD_DP, COV_DP, BD_EOP). "
                             "if none is selected no fairness criterion is applied")
    parser.add_argument('-fv', '--fairness_value', type=float, required=False, help='the value of lambda')
    parser.add_argument('-flr', '--fairness_learning_rate', type=float, required=False,
                        help="define the learning rate of lambda")
    parser.add_argument('-fbs', '--fairness_batch_size', type=int, required=False,
                        help='batch size to be used to learn lambda')
    parser.add_argument('-fe', '--fairness_epochs', type=int, required=False,
                        help='number of epochs to be used to learn lambda')
    parser.add_argument('-faug', '--fairness_augmented', required=False, action='store_true')

    args = parser.parse_args()

    if args.fairness_type is not None and args.fairness_value is None:
        parser.error('when using --fairness_type, --fairness_value has to be specified')

    if args.fairness_type is not None and \
            ((args.fairness_epochs is None or
              args.fairness_learning_rate is None or
              args.fairness_batch_size is None) and not
             (args.fairness_epochs is None and
              args.fairness_learning_rate is None and
              args.fairness_batch_size is None)):
        parser.error('--fairness_epochs, --fairness_learning_rate, fairness_batch_size and '
                     'have to be fully specified or not specified at all')
    single_run(args)
