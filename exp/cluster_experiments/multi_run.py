import argparse
import subprocess
from copy import deepcopy

import numpy as np


def multi_run(args):
    if args.fairness_type is not None:
        base_path = "{}/{}/{}".format(args.path, args.data, args.fairness_type)
        if args.fairness_lower_bound is None and args.fairness_values is None:
            parser.error(
                'when using --fairness_type, either --fairness_lower_bound or --fairness_values has to be specified')
        elif args.fairness_values is not None:
            lambdas = args.fairness_values
        elif args.fairness_upper_bound is not None:
            fairness_num = args.fairness_number if args.fairness_number is not None else 20
            lambdas = np.geomspace(args.fairness_lower_bound, args.fairness_upper_bound, endpoint=True,
                                   num=fairness_num)
        else:
            lambdas = args.fairness_lower_bound
    else:
        base_path = "{}/{}".format(args.path, args.data)

    for cost in args.costs:
        for learning_rate in args.learning_rates:
            for time_steps in args.time_steps:
                for epochs in args.epochs:
                    for batch_size in args.batch_sizes:
                        for num_batches in args.num_batches:
                            command = ["python", "run.py",
                                       "-d", str(args.data),
                                       "-c", str(cost),
                                       "-lr", str(learning_rate),
                                       "-i", str(args.iterations),
                                       "-p", "{}/raw".format(base_path),
                                       "-ts", str(time_steps),
                                       "-e", str(epochs),
                                       "-bs", str(batch_size),
                                       "-nb", str(num_batches)]
                            if args.asynchronous:
                                command.append("-a")
                            if args.plot:
                                command.append("--plot")

                            if args.fairness_type is not None:
                                for fairness_rate in lambdas:
                                    temp_command = deepcopy(command)
                                    temp_command.extend(["-f", str(args.fairness_type), "-fv", str(fairness_rate)])
                                    subprocess.run(temp_command)
                            else:
                                subprocess.run(command)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--data', type=str, required=True,
                        help="select the distribution (FICO, COMPAS, ADULT, GERMAN)")

    parser.add_argument('-p', '--path', type=str, required=False, help="save path for the results")
    parser.add_argument('-c', '--costs', type=float, nargs='+', required=True, help="define the utility cost c")
    parser.add_argument('-lr', '--learning_rates', type=float, nargs='+', required=True,
                        help="define the learning rate of theta")
    parser.add_argument('-ts', '--time_steps', type=int, nargs='+', required=True, help='list of time steps to be used')
    parser.add_argument('-e', '--epochs', type=int, nargs='+', required=True, help='list of epochs to be used')
    parser.add_argument('-bs', '--batch_sizes', type=int, nargs='+', required=True,
                        help='list of batch sizes to be used')
    parser.add_argument('-nb', '--num_batches', type=int, nargs='+', required=True,
                        help='list of number of batches to be used')

    parser.add_argument('-i', '--iterations', type=int, required=True, help='the number of internal iterations')
    parser.add_argument('-a', '--asynchronous', action='store_true')
    parser.add_argument('--plot', required=False, action='store_true')

    parser.add_argument('-f', '--fairness_type', type=str, required=False,
                        help="select the type of fairness (BD_DP, COV_DP, BP_EOP). "
                             "if none is selected no fairness criterion is applied")

    parser.add_argument('-fv', '--fairness_values', type=float, nargs='+', required=False,
                        help='list of fairness values to be used')

    parser.add_argument('-fl', '--fairness_lower_bound', type=float, required=False, help='the lowest value for lambda')
    parser.add_argument('-fu', '--fairness_upper_bound', type=float, required=False,
                        help='the highest value for lambda')
    parser.add_argument('-fn', '--fairness_number', type=int, required=False,
                        help='the number of lambda values tested in the range')

    args = parser.parse_args()
    multi_run(args)
