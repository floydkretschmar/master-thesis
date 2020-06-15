import argparse
import os
import sys

import numpy as np
import torch

module_path = os.path.abspath(os.path.join("../.."))
if module_path not in sys.path:
    sys.path.append(module_path)

import src.util as util
from src.feature_map import IdentityFeatureMap
from src.functions import cost_utility, cost_utility_gradient, cost_utility_probability
from src.plotting import plot_mean, plot_median
from src.training import train
from src.training_evaluation import UTILITY, COVARIANCE_OF_DECISION_DP
from src.distribution import FICODistribution, COMPASDistribution, AdultCreditDistribution, GermanCreditDistribution
from src.util import mean_difference, get_list_of_seeds, mean
from src.optimization import PenaltyOptimizationTarget, LagrangianOptimizationTarget, \
    AugmentedLagrangianOptimizationTarget, ManualGradientPenaltyOptimizationTarget, \
    ManualGradientAugmentedLagrangianOptimizationTarget, ManualGradientLagrangianOptimizationTarget
from src.policy import LogisticPolicy, NeuralNetworkPolicy


# region Fairness Definitions

def calc_benefit(decisions, ips_weights):
    if ips_weights is not None:
        decisions = decisions * ips_weights

    return decisions


def calc_covariance(s, decisions, ips_weights):
    new_s = 1 - (2 * s)

    if ips_weights is not None:
        mu_s = mean(new_s * ips_weights, axis=0)
        d = decisions * ips_weights
    else:
        mu_s = mean(new_s, axis=0)
        d = decisions

    covariance = (new_s - mu_s) * d
    return covariance


def fairness_function_gradient(**fairness_kwargs):
    policy = fairness_kwargs["policy"]
    x = fairness_kwargs["x"]
    s = fairness_kwargs["s"]
    y = fairness_kwargs["y"]
    decisions = fairness_kwargs["decisions"]
    ips_weights = fairness_kwargs["ips_weights"]

    if args.fairness_type == "BD_DP" or args.fairness_type == "BD_EOP":
        result = calc_benefit(decisions, ips_weights)
    elif args.fairness_type == "COV_DP":
        result = calc_covariance(s, decisions, ips_weights)

    log_gradient = policy.log_policy_gradient(x, s)
    grad = log_gradient * result

    if args.fairness_type == "BD_DP":
        return mean_difference(grad, s)
    elif args.fairness_type == "COV_DP":
        return mean(grad, axis=0)
    elif args.fairness_type == "BD_EOP":
        y1_indices = np.where(y == 1)
        return mean_difference(grad[y1_indices], s[y1_indices])


def fairness_function(type=None, **fairness_kwargs):
    s = fairness_kwargs["s"]
    ips_weights = fairness_kwargs["ips_weights"] if "ips_weights" in fairness_kwargs else None
    decisions = "decision_probabilities" not in fairness_kwargs or not (args.policy_type == "NN" and ips_weights is None)

    decisions = fairness_kwargs["decisions"] if decisions else fairness_kwargs["decision_probabilities"]
    y = fairness_kwargs["y"]

    type = args.fairness_type if type is None else type

    if type == "BD_DP":
        benefit = calc_benefit(decisions, ips_weights)
        return mean_difference(benefit, s)
    elif type == "COV_DP":
        covariance = calc_covariance(s, decisions, ips_weights)
        return mean(covariance, axis=0)
    elif type == "BD_EOP":
        benefit = calc_benefit(decisions, ips_weights)
        y1_indices = np.where(y.squeeze() == 1)
        return mean_difference(benefit[y1_indices], s[y1_indices])


def no_fairness(**fairness_kwargs):
    return 0.0


def covariance_of_decision(s, y, decisions):
    return fairness_function(
        type="COV_DP",
        x=None,
        s=s,
        y=y,
        decisions=decisions,
        ips_weights=None,
        policy=None)


def utility(**util_params):
    return cost_utility(cost_factor=args.cost, **util_params)


def utility_gradient(**util_params):
    return cost_utility_gradient(cost_factor=args.cost, **util_params)


def utility_nn(**util_params):
    return cost_utility_probability(cost_factor=args.cost, **util_params)


# endregion
def _build_optimization_target(args):
    neural_network = args.policy_type == "NN"

    additional_args = {}
    if args.fairness_type is not None and args.fairness_learning_rate is not None:
        if not args.fairness_augmented:
            optim_target_constructor = LagrangianOptimizationTarget if neural_network \
                else ManualGradientLagrangianOptimizationTarget
        else:
            optim_target_constructor = AugmentedLagrangianOptimizationTarget if neural_network \
                else ManualGradientAugmentedLagrangianOptimizationTarget
            additional_args["penalty_constant"] = args.fairness_learning_rate
    else:
        optim_target_constructor = PenaltyOptimizationTarget if neural_network \
            else ManualGradientPenaltyOptimizationTarget

    if args.fairness_type is None:
        initial_fairness = 0.0
        fair_fct = no_fairness
        fair_fct_grad = no_fairness
    else:
        initial_fairness = args.fairness_value
        fair_fct = fairness_function
        fair_fct_grad = fairness_function_gradient

    if not neural_network:
        return optim_target_constructor(initial_fairness,
                                        utility,
                                        utility_gradient,
                                        fair_fct,
                                        fair_fct_grad,
                                        **additional_args), initial_fairness
    else:
        return optim_target_constructor(initial_fairness,
                                        utility_nn,
                                        fair_fct,
                                        **additional_args), initial_fairness


def single_run(args):
    if args.data == "FICO":
        distibution = FICODistribution(bias=True, fraction_protected=0.5)
    elif args.data == "COMPAS":
        distibution = COMPASDistribution(bias=True, test_percentage=0.2)
    elif args.data == "ADULT":
        distibution = AdultCreditDistribution(bias=True, test_percentage=0.2)
    elif args.data == "GERMAN":
        distibution = GermanCreditDistribution(bias=True, test_percentage=0.3)

    if args.policy_type == "LOG":
        model = LogisticPolicy(np.zeros((distibution.feature_dimension)),
                               IdentityFeatureMap(distibution.feature_dimension),
                               False)
    elif args.policy_type == "NN":
        model = NeuralNetworkPolicy(distibution.feature_dimension, False)

    optimization_target, initial_lambda = _build_optimization_target(args)

    training_parameters = {
        "model": model,
        "distribution": distibution,
        "optimization_target": optimization_target,
        "parameter_optimization": {
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "learn_on_entire_history": False,
            "time_steps": args.time_steps,
            "clip_weights": args.ip_weight_clipping,
            "change_percentage": args.change_percentage,
            "change_iterations": args.change_iterations
        },
        "data": {
            "num_train_samples": args.num_samples,
            "num_test_samples": 10000,
            "fix_seeds": True
        },
        "evaluation": {
            UTILITY: {
                "measure_function": utility,
                "detailed": False
            },
            COVARIANCE_OF_DECISION_DP: {
                "measure_function": covariance_of_decision,
                "detailed": False
            }
        }
    }

    if args.fairness_type is not None and args.fairness_learning_rate is not None:
        training_parameters["lagrangian_optimization"] = {
            "epochs": args.fairness_epochs,
            "batch_size": args.fairness_batch_size,
            "learning_rate": args.fairness_learning_rate
        }

    if args.seed_path:
        del training_parameters["data"]["fix_seeds"]

        if os.path.isfile(args.seed_path):
            seeds = np.load(args.seed_path)
            training_parameters["data"]["training_seeds"] = seeds["train"]
            training_parameters["data"]["test_seed"] = seeds["test"]
        else:
            train_seeds = get_list_of_seeds(200)
            test_seeds = get_list_of_seeds(1)
            training_parameters["data"]["training_seeds"] = train_seeds
            training_parameters["data"]["test_seed"] = test_seeds
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

    parser.add_argument("-sp", "--seed_path", type=str, required=False, help="path for the seeds .npz file")

    parser.add_argument("-d", "--data", type=str, required=True,
                        help="select the distribution (FICO, COMPAS, ADULT, GERMAN)")
    parser.add_argument("-c", "--cost", type=float, required=True, help="define the utility cost c")
    parser.add_argument("-lr", "--learning_rate", type=float, required=True, help="define the learning rate of theta")
    parser.add_argument("-p", "--path", type=str, required=False, help="save path for the result")
    parser.add_argument("-pt", "--policy_type", type=str, required=False, default="LOG",
                        help="(NN, LOG), default = LOG")
    parser.add_argument("-ts", "--time_steps", type=int, required=True, help="number of time steps to be used")
    parser.add_argument("-e", "--epochs", type=int, required=True, help="number of epochs to be used")
    parser.add_argument("-bs", "--batch_size", type=int, required=True, help="batch size to be used")
    parser.add_argument("-ci", "--change_iterations", type=int, required=False, default=5,
                        help="the number of iterations without the amout of percentage improvemnt specified by "
                             "--change_percentage after which the training of the policy will be stopped."
                             "(Default = 5)")
    parser.add_argument("-cp", "--change_percentage", type=int, required=False, default=0.05,
                        help="the percentage of improvement per training epoch that is considered the minimum amount of"
                             "improvement. (Default = 0.05)")
    parser.add_argument("-ns", "--num_samples", type=int, required=True, help="number of batches to be used")
    parser.add_argument("-i", "--iterations", type=int, required=True, help="the number of internal iterations")
    parser.add_argument("-ipc", "--ip_weight_clipping", action="store_true")
    parser.add_argument("-a", "--asynchronous", action="store_true")
    parser.add_argument("--plot", required=False, action="store_true")
    parser.add_argument("-pid", "--process_id", type=str, required=False, help="process id for identification")

    parser.add_argument("-f", "--fairness_type", type=str, required=False,
                        help="select the type of fairness (BD_DP, COV_DP, BD_EOP). "
                             "if none is selected no fairness criterion is applied")
    parser.add_argument("-fv", "--fairness_value", type=float, required=False, help="the value of lambda")
    parser.add_argument("-flr", "--fairness_learning_rate", type=float, required=False,
                        help="define the learning rate of lambda")
    parser.add_argument("-fbs", "--fairness_batch_size", type=int, required=False,
                        help="batch size to be used to learn lambda")
    parser.add_argument("-fe", "--fairness_epochs", type=int, required=False,
                        help="number of epochs to be used to learn lambda")
    parser.add_argument("-faug", "--fairness_augmented", required=False, action="store_true")

    parser.add_argument("--CUDA", required=False, action="store_true")

    args = parser.parse_args()

    if args.CUDA:
        util.CUDA = True

    if args.fairness_type is not None and args.fairness_value is None:
        parser.error("when using --fairness_type, --fairness_value has to be specified")

    if args.fairness_type is not None and \
            ((args.fairness_epochs is None or
              args.fairness_learning_rate is None or
              args.fairness_batch_size is None) and not
             (args.fairness_epochs is None and
              args.fairness_learning_rate is None and
              args.fairness_batch_size is None)):
        parser.error("--fairness_epochs, --fairness_learning_rate, fairness_batch_size and "
                     "have to be fully specified or not specified at all")
    single_run(args)
