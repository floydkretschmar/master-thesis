import subprocess
import argparse
import os
import sys

from copy import deepcopy
import numpy as np

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

module_path = os.path.abspath(os.path.join("../.."))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.util import get_list_of_seeds

def _fairness_extensions(args, fairness_rates, build=False):
    extensions = []
    for fairness_rate in fairness_rates:
        if build:
            extension = "-f {} -fv {}".format(args.fairness_type, fairness_rate)
        else:
            extension = ["-f", str(args.fairness_type), "-fv", str(fairness_rate)]

        if args.fairness_learning_rates is not None:
            for learning_rate in args.fairness_learning_rates:
                for batch_size in args.fairness_batch_sizes:
                    for epochs in args.fairness_epochs:
                        if build:
                            extensions.append("{} -flr {} -fbs {} -fe {}".format(extension,
                                                                                 learning_rate,
                                                                                 batch_size,
                                                                                 epochs))
                        else:
                            temp_extension = deepcopy(extension)
                            temp_extension.extend(["-flr", str(learning_rate),
                                                   "-fbs", str(batch_size),
                                                   "-fe", str(epochs)])
                            extensions.append(temp_extension)
        else:
            extensions.append(extension)

    return extensions


def _build_submit_file(args, base_path, lambdas, seeds):
    sub_file_name = args.file_path
    print("## Started building {} ##".format(sub_file_name))

    with open(sub_file_name, "w") as file:
        file.write("# ----------------------------------------------------------------------- #\n")
        file.write("# RUNTIME LIMITATION                                                      #\n")
        file.write("# ----------------------------------------------------------------------- #\n\n")
        file.write("# Maximum expected execution time for the job, in seconds\n")
        file.write("# 43200 = 12h\n")
        file.write("# 86400 = 24h\n")
        file.write("MaxTime = 43200\n\n")
        file.write("# Kill the jobs without warning\n")
        file.write("periodic_remove = (JobStatus =?= 2) && ((CurrentTime - JobCurrentStartDate) >= $(MaxTime))\n\n")
        file.write("# ----------------------------------------------------------------------- #\n")
        file.write("# RESSOURCE SELECTION                                                     #\n")
        file.write("# ----------------------------------------------------------------------- #\n\n")
        file.write("request_memory = {}\n".format(args.ram))
        file.write("request_cpus = {}\n\n".format(args.cpu))
        file.write("# ----------------------------------------------------------------------- #\n")
        file.write("# FOLDER SELECTION                                                        #\n")
        file.write("# ----------------------------------------------------------------------- #\n\n")
        file.write("environment = \"PYTHONUNBUFFERED=TRUE\"\n")
        file.write("executable = {}\n\n".format(args.python_path))
        file.write("error = {}/error/experiment.$(Process).err\n".format(base_path))
        file.write("output = {}/output/experiment.$(Process).out\n".format(base_path))
        file.write("log = {}/log/experiment.$(Process).log\n".format(base_path))
        file.write("# ----------------------------------------------------------------------- #\n")
        file.write("# QUEUE                                                                   #\n")
        file.write("# ----------------------------------------------------------------------- #\n\n")

        for cost in args.costs:
            for learning_rate in args.learning_rates:
                for time_steps in args.time_steps:
                    for epochs in args.epochs:
                        for batch_size in args.batch_sizes:
                            for history_learning in args.history_learning:
                                command = "run.py " \
                                          "-d {} " \
                                          "-c {} " \
                                          "-lr {} " \
                                          "-p {}" \
                                          "-ts {} " \
                                          "-e {} " \
                                          "-bs {} " \
                                          "-ns {} " \
                                          "-ns_t {} " \
                                          "{}" \
                                          "{}" \
                                          "{}" \
                                          "{}" \
                                          "{}" \
                                          "{}" \
                                          "{}" \
                                          "{}" \
                                          "{}" \
                                          "{}".format(args.data,
                                                      cost,
                                                      learning_rate,
                                                      "{}/raw ".format(base_path),
                                                      time_steps,
                                                      epochs,
                                                      batch_size,
                                                      args.num_samples,
                                                      args.num_samples_test,
                                                      "-ci {} ".format(
                                                          args.change_iterations) if args.change_iterations else "",
                                                      "-cp {} ".format(
                                                          args.change_percentage) if args.change_percentage else "",
                                                      "-pt {} ".format(args.policy_type) if args.policy_type else "",
                                                      "-fd {} ".format(args.fairness_delta) if args.fairness_delta else "",
                                                      "-a " if args.asynchronous else "",
                                                      "--plot " if args.plot else "",
                                                      "-ipc " if args.ip_weight_clipping else "",
                                                      "-hl " if history_learning else "",
                                                      "-faug " if args.fairness_augmented else "",
                                                      "-pid $(Process)" if args.iterations else "")

                                if args.fairness_type is not None:
                                    fairness_extensions = [extension for extension in _fairness_extensions(args, lambdas, build=True)]
                                else:
                                    fairness_extensions = [""]

                                for extension in fairness_extensions:
                                    args_str = "arguments = {} {}".format(command, extension)

                                    if seeds is None:
                                        file.write("{}\n".format(args_str))
                                        file.write("queue {}\n".format(args.iterations if args.iterations is not None else ""))
                                    else:
                                        for seed in seeds:
                                            seed_args = "{} -s {}\n".format(args_str, seed)
                                            file.write(seed_args)
                                            file.write("queue\n")

    print("## FInished building {} ##".format(sub_file_name))


def _multi_run(args, base_path, lambdas, seeds):
    for cost in args.costs:
        for learning_rate in args.learning_rates:
            for time_steps in args.time_steps:
                for epochs in args.epochs:
                    for batch_size in args.batch_sizes:
                        for history_learning in args.history_learning:
                            command = ["python", "run.py",
                                       "-d", str(args.data),
                                       "-c", str(cost),
                                       "-lr", str(learning_rate),
                                       "-p", "{}/raw".format(base_path),
                                       "-ts", str(time_steps),
                                       "-e", str(epochs),
                                       "-bs", str(batch_size),
                                       "-ns", str(args.num_samples),
                                       "-ns_t", str(args.num_samples_test)]
                            if args.asynchronous:
                                command.append("-a")
                            if history_learning:
                                command.append("-hl")
                            if args.plot:
                                command.append("--plot")
                            if args.ip_weight_clipping:
                                command.append("-ipc")
                            if args.fairness_augmented:
                                command.append("-faug")
                            if args.change_iterations:
                                command.extend(["-ci", str(args.change_iterations)])
                            if args.change_percentage:
                                command.extend(["-cp", str(args.change_percentage)])
                            if args.policy_type:
                                command.extend(["-pt", str(args.policy_type)])
                            if args.fairness_delta:
                                command.extend(["-fd", str(args.fairness_delta)])

                            if args.fairness_type is not None:
                                fairness_extensions = [extension for extension in
                                                       _fairness_extensions(args, lambdas, build=False)]
                            else:
                                fairness_extensions = []

                            for extension in fairness_extensions:
                                temp_command = deepcopy(command)
                                temp_command.extend(extension)

                                for iter in range(args.iterations):
                                    if seeds is None:
                                        subprocess.run(temp_command)
                                    else:
                                        seed = seeds[iter]
                                        temp_command.extend(["-s", str(seed)])
                                        subprocess.run(temp_command)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Configuration parameters
    parser.add_argument('-d', '--data', type=str, required=True,
                        help="select the distribution (FICO, COMPAS, ADULT, GERMAN)")
    parser.add_argument('-p', '--path', type=str, required=False, help="save path for the results")
    parser.add_argument('-sp', '--seed_path', type=str, required=False, help="path for the.npz file storing the seeds")

    # Policy training parameters
    parser.add_argument("-pt", "--policy_type", type=str, required=False, default="LOG", help="(NN, LOG), default = LOG")
    parser.add_argument('-c', '--costs', type=float, nargs='+', required=True, help="define the utility cost c")
    parser.add_argument('-lr', '--learning_rates', type=float, nargs='+', required=True,
                        help="define the learning rate of theta")
    parser.add_argument('-ts', '--time_steps', type=int, nargs='+', required=True, help='list of time steps to be used')
    parser.add_argument('-e', '--epochs', type=int, nargs='+', required=True, help='list of epochs to be used')
    parser.add_argument('-bs', '--batch_sizes', type=int, nargs='+', required=True,
                        help='list of batch sizes to be used')
    parser.add_argument('-ci', '--change_iterations', type=int, required=False,
                        help='the number of iterations without the amout of percentage improvemnt specified by '
                             '--change_percentage after which the training of the policy will be stopped.')
    parser.add_argument('-cp', '--change_percentage', type=int, required=False,
                        help='the percentage of improvement per training epoch that is considered the minimum amount of'
                             'improvement. ')
    parser.add_argument('-ns', '--num_samples', type=int, required=True,
                        help='list of number of samples to drawn each time step')
    parser.add_argument("-ns_t", "--num_samples_test", type=int, required=True, help="number of test samples")
    parser.add_argument('-i', '--iterations', type=int, required=True, help='the number of iterations')
    parser.add_argument('-hl', '--history_learning', type=str2bool, nargs='+', required=True)

    # ip weighting parameters
    parser.add_argument('-ipc', '--ip_weight_clipping', action='store_true')

    # technical parameters
    parser.add_argument('-a', '--asynchronous', action='store_true')
    parser.add_argument('--plot', required=False, action='store_true')

    # Fairness parameters
    parser.add_argument('-f', '--fairness_type', type=str, required=False,
                        help="select the type of fairness (BD_DP, COV_DP, BP_EOP). "
                             "if none is selected no fairness criterion is applied")

    parser.add_argument('-fv', '--fairness_values', type=float, nargs='+', required=False,
                        help='list of fairness values to be used')

    parser.add_argument('-fl', '--fairness_lower_bound', type=float, required=False, help='the lowest value for lambda')
    parser.add_argument('-fu', '--fairness_upper_bound', type=float, required=False,
                        help='the highest value for lambda')
    parser.add_argument('-fn', '--fairness_number', type=int, required=False, default=-1,
                        help='the number of lambda values tested in the range.')

    parser.add_argument('-flr', '--fairness_learning_rates', type=float, required=False, nargs='+',
                        help="define the learning rates of lambda")
    parser.add_argument('-fbs', '--fairness_batch_sizes', type=int, required=False, nargs='+',
                        help='batch sizes to be used to learn lambda')
    parser.add_argument('-fe', '--fairness_epochs', type=int, required=False, nargs='+',
                        help='number of epochs to be used to learn lambda')
    parser.add_argument('-faug', '--fairness_augmented', required=False, action='store_true')
    parser.add_argument('-fd', '--fairness_delta', type=float, required=False)

    # Build script parameters
    parser.add_argument('--build_submit', required=False, action='store_true')
    parser.add_argument('--file_path', type=str, required=False, help="path and name of the submit file that will be created.")
    parser.add_argument('-pp', '--python_path', type=str, required=False, help="path of the python executable")
    parser.add_argument('--ram', type=int, required=False, help='the RAM requested (default = 6144)', default=6144)
    parser.add_argument('--cpu', type=int, required=False, help='the number of CPUs requested (default = 1)', default=1)

    args = parser.parse_args()

    if args.build_submit and args.python_path is None:
        parser.error('when using --build_submit, --python_path has to be specified')
    if args.build_submit and args.file_path is None:
        parser.error('when using --build_submit, --file_path has to be specified')

    if args.seed_path:
        if os.path.isfile(args.seed_path):
            seeds = np.load(args.seed_path)
            if len(seeds) != args.iterations:
                raise TypeError("The specified seed file {} has a different number of seeds than the number of "
                                "iterations {} specified by -i.".format(args.seed_path, args.iterations))
        else:
            seeds = get_list_of_seeds(args.iterations)
            np.save(args.seed_path, seeds)
    else:
        seeds = None

    num_fairness_batches = None
    if args.fairness_type is not None:
        base_path = "{}/{}/{}/{}".format(args.path, args.data, args.policy_type, args.fairness_type)
        if (args.fairness_lower_bound is None and args.fairness_upper_bound is not None) or \
                (args.fairness_lower_bound is not None and args.fairness_upper_bound is None):
            parser.error('--fairness_lower_bound and --fairness_upper_bound have to be specified together')
        elif args.fairness_type is not None and \
                ((args.fairness_epochs is None or
                  args.fairness_learning_rates is None or
                  args.fairness_batch_sizes is None) and not
                 (args.fairness_epochs is None and
                  args.fairness_learning_rates is None and
                  args.fairness_batch_sizes is None)):
            parser.error(
                '--fairness_epochs, --fairness_learning_rates, fairness_batch_sizes and '
                'have to be fully specified or not specified at all')
        elif args.fairness_values is not None:
            lambdas = args.fairness_values
        elif args.fairness_lower_bound is not None and args.fairness_upper_bound is not None:
            fairness_num = args.fairness_number
            if fairness_num != -1:
                lambdas = np.geomspace(args.fairness_lower_bound,
                                       args.fairness_upper_bound,
                                       endpoint=True,
                                       num=fairness_num)
            else:
                current_value = args.fairness_lower_bound
                power = np.log10(current_value)
                lambdas = []

                while power + 1 <= np.log10(args.fairness_upper_bound):
                    lambdas.extend(np.linspace(np.power(10, power), np.power(10, power + 1), num=10).tolist())
                    power += 1
                lambdas = np.sort(np.unique(np.array(lambdas, dtype=float)))
        else:
            lambdas = [0.0]
    else:
        base_path = "{}/{}/{}".format(args.path, args.data, args.policy_type)
        lambdas = [0.0]

    if args.build_submit:
        _build_submit_file(args, base_path, lambdas, seeds)
    else:
        _multi_run(args, base_path, lambdas, seeds)
