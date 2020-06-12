import argparse
import subprocess
from copy import deepcopy

import numpy as np


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


def _build_submit_file(args, base_path, lambdas):
    base_name = args.data if not args.file_name else args.file_name
    sub_file_name = "./{}.sub".format(base_name) if args.fairness_type is None else "./{}_{}.sub".format(base_name,
                                                                                                         args.fairness_type)

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
                            command = "run.py " \
                                      "-d {} " \
                                      "-c {} " \
                                      "-lr {} " \
                                      "-i {} " \
                                      "-p {}" \
                                      "-ts {} " \
                                      "-e {} " \
                                      "-bs {} " \
                                      "-ns {} " \
                                      "{} " \
                                      "{} " \
                                      "{} " \
                                      "{} " \
                                      "{} " \
                                      "{} " \
                                      "{} " \
                                      "{} " \
                                      "{}".format(args.data,
                                                  cost,
                                                  learning_rate,
                                                  args.iterations,
                                                  "{}/raw ".format(base_path),
                                                  time_steps,
                                                  epochs,
                                                  batch_size,
                                                  args.num_samples,
                                                  "-sp {}".format(args.seed_path) if args.seed_path else "",
                                                  "-ci {}".format(
                                                      args.change_iterations) if args.change_iterations else "",
                                                  "-cp {}".format(
                                                      args.change_percentage) if args.change_percentage else "",
                                                  "-pt {}".format(args.policy_type) if args.policy_type else "",
                                                  "-a " if args.asynchronous else "",
                                                  "--plot " if args.plot else "",
                                                  "-ipc " if args.ip_weight_clipping else "",
                                                  "-faug" if args.fairness_augmented else "",
                                                  "-pid $(Process)" if args.queue_num else "")

                            if args.fairness_type is not None:
                                for extension in _fairness_extensions(args, lambdas, build=True):
                                    file.write("arguments = {} {}\n".format(command, extension))
                                    file.write("queue {}\n".format(args.queue_num
                                                                   if args.queue_num is not None else ""))
                            else:
                                file.write("arguments = {}\n".format(command))
                                file.write("queue {}\n".format(args.queue_num
                                                               if args.queue_num is not None else ""))

    print("## FInished building {} ##".format(sub_file_name))


def _multi_run(args, base_path, lambdas):
    for cost in args.costs:
        for learning_rate in args.learning_rates:
            for time_steps in args.time_steps:
                for epochs in args.epochs:
                    for batch_size in args.batch_sizes:
                        command = ["python", "run.py",
                                   "-d", str(args.data),
                                   "-c", str(cost),
                                   "-lr", str(learning_rate),
                                   "-i", str(args.iterations),
                                   "-p", "{}/raw".format(base_path),
                                   "-ts", str(time_steps),
                                   "-e", str(epochs),
                                   "-bs", str(batch_size),
                                   "-ns", str(args.num_samples)]
                        if args.asynchronous:
                            command.append("-a")
                        if args.plot:
                            command.append("--plot")
                        if args.ip_weight_clipping:
                            command.append("-ipc")
                        if args.fairness_augmented:
                            command.append("-faug")
                        if args.seed_path:
                            command.extend(["-sp", args.seed_path])
                        if args.change_iterations:
                            command.extend(["-ci", str(args.change_iterations)])
                        if args.change_percentage:
                            command.extend(["-cp", str(args.change_percentage)])
                        if args.policy_type:
                            command.extend(["-pt", str(args.policy_type)])

                        if args.fairness_type is not None:
                            for extension in _fairness_extensions(args, lambdas, build=False):
                                temp_command = deepcopy(command)
                                temp_command.extend(extension)
                                subprocess.run(temp_command)
                        else:
                            subprocess.run(command)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Configuration parameters
    parser.add_argument('-d', '--data', type=str, required=True,
                        help="select the distribution (FICO, COMPAS, ADULT, GERMAN)")
    parser.add_argument('-p', '--path', type=str, required=False, help="save path for the results")
    parser.add_argument('-sp', '--seed_path', type=str, required=False, help="path for the seeds .npz file")

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
                        help='list of number of batches to be used')
    parser.add_argument('-i', '--iterations', type=int, required=True, help='the number of internal iterations')

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

    parser.add_argument('-fi', '--fairness_iterations', type=int, nargs='+', required=False,
                        help='number of iterations that the dual gradient loop will be repeated')
    parser.add_argument('-flr', '--fairness_learning_rates', type=float, required=False, nargs='+',
                        help="define the learning rates of lambda")
    parser.add_argument('-fbs', '--fairness_batch_sizes', type=int, required=False, nargs='+',
                        help='batch sizes to be used to learn lambda')
    parser.add_argument('-fe', '--fairness_epochs', type=int, required=False, nargs='+',
                        help='number of epochs to be used to learn lambda')
    parser.add_argument('-faug', '--fairness_augmented', required=False, action='store_true')

    # Build script parameters
    parser.add_argument('--build_submit', required=False, action='store_true')
    parser.add_argument('--file_name', type=str, required=False, help="name of the submit file")
    parser.add_argument('-pp', '--python_path', type=str, required=False, help="path of the python executable")
    parser.add_argument('-q', '--queue_num', type=int, required=False,
                        help="the number of process that should be queued")
    parser.add_argument('--ram', type=int, required=False, help='the RAM requested (default = 6144)', default=6144)
    parser.add_argument('--cpu', type=int, required=False, help='the number of CPUs requested (default = 1)', default=1)

    args = parser.parse_args()

    if args.build_submit and args.python_path is None:
        parser.error('when using --build_submit, --python_path has to be specified')

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
        _build_submit_file(args, base_path, lambdas)
    else:
        _multi_run(args, base_path, lambdas)
