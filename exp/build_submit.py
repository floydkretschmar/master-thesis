import argparse

import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('-d', '--data', type=str, required=True, help="select the distribution (FICO, COMPAS, ADULT)")
parser.add_argument('-op', '--output_path', type=str, required=False, help="output path for the submission")
parser.add_argument('-lp', '--log_path', type=str, required=False, help="log path for the submission")
parser.add_argument('-ep', '--error_path', type=str, required=False, help="error path for the submission")
parser.add_argument('-pp', '--python_path', type=str, required=False, help="path of the python executable")
parser.add_argument('-p', '--path', type=str, required=False, help="save path for the results")

parser.add_argument('-c', '--costs', type=float, nargs='+', required=True, help="define the utility cost c")
parser.add_argument('-lr', '--learning_rates', type=float, nargs='+', required=True,
                    help="define the learning rate of theta")
parser.add_argument('-ts', '--time_steps', type=int, nargs='+', required=True, help='list of time steps to be used')
parser.add_argument('-e', '--epochs', type=int, nargs='+', required=True, help='list of epochs to be used')
parser.add_argument('-bs', '--batch_sizes', type=int, nargs='+', required=True, help='list of batch sizes to be used')
parser.add_argument('-nb', '--num_batches', type=int, nargs='+', required=True,
                    help='list of number of batches to be used')

parser.add_argument('-i', '--iterations', type=int, required=True, help='the number of internal iterations')
parser.add_argument('-a', '--asynchronous', action='store_true')
parser.add_argument('--plot', required=False, action='store_true')

parser.add_argument('-f', '--fairness_type', type=str, required=False,
                    help="select the type of fairness (BD_DP, COV_DP, BP_EOP). "
                         "if none is selected no fairness criterion is applied")
parser.add_argument('-fl', '--fairness_lower_bound', type=float, required=False, help='the lowest value for lambda')
parser.add_argument('-fu', '--fairness_upper_bound', type=float, required=False, help='the highest value for lambda')
parser.add_argument('-fn', '--fairness_number', type=int, required=False,
                    help='the number of lambda values tested in the range')

args = parser.parse_args()

if args.fairness_type is not None:
    if args.fairness_lower_bound is None:
        parser.error('when using --fairness_type, at least --fairness_lower_bound has to be specified')
    elif args.fairness_upper_bound is not None:
        fairness_num = args.fairness_number if args.fairness_number is not None else 20
        lambdas = np.geomspace(args.fairness_lower_bound, args.fairness_upper_bound, endpoint=True,
                               num=fairness_num)
    else:
        lambdas = args.fairness_lower_bound

sub_file_name = "./{}.sub".format(args.data) if args.fairness_type is None else "./{}_{}.sub".format(args.data,
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
    file.write("request_memory = 1024\n")
    file.write("request_cpus = 1\n\n")
    file.write("# ----------------------------------------------------------------------- #\n")
    file.write("# FOLDER SELECTION                                                        #\n")
    file.write("# ----------------------------------------------------------------------- #\n\n")
    file.write("environment = \"PYTHONUNBUFFERED=TRUE\"\n")
    file.write("executable = {}\n\n".format(args.python_path))
    file.write("error = {}/experiment.$(Process).err\n".format(args.error_path))
    file.write("output = {}/experiment.$(Process).out\n".format(args.output_path))
    file.write("log = {}/experiment.$(Process).log\n\n".format(args.log_path))
    file.write("# ----------------------------------------------------------------------- #\n")
    file.write("# QUEUE                                                                   #\n")
    file.write("# ----------------------------------------------------------------------- #\n\n")

    for cost in args.costs:
        for learning_rate in args.learning_rates:
            for time_steps in args.time_steps:
                for epochs in args.epochs:
                    for batch_size in args.batch_sizes:
                        for num_batches in args.num_batches:
                            command = "run.py " \
                                      "-d {} " \
                                      "-c {} " \
                                      "-lr {} " \
                                      "{} " \
                                      "-i 1 " \
                                      "-p {}/{}/raw " \
                                      "-ts {} " \
                                      "-e {} " \
                                      "-bs {} " \
                                      "-nb {} " \
                                      "-pid $(Process)".format(args.data,
                                                               cost,
                                                               learning_rate,
                                                               "-a " if args.asynchronous else "",
                                                               args.path, args.data,
                                                               time_steps,
                                                               epochs,
                                                               batch_size,
                                                               num_batches)

                            commands = []
                            if args.fairness_type is not None:
                                for fairness_rate in lambdas:
                                    commands.append(
                                        "{} -f {} -fv {}".format(command, args.fairness_type, fairness_rate))
                            else:
                                commands.append(command)

                            for command in commands:
                                file.write("arguments = {}\n".format(command))
                                file.write("queue {}\n".format(args.iterations))

print("## Finished building {} ##".format(sub_file_name))
