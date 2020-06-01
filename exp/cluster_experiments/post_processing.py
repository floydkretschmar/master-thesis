import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

module_path = os.path.abspath(os.path.join('../..'))

if module_path not in sys.path:
    sys.path.append(module_path)

from src.training_evaluation import ModelParameters, MultiStatistics, UTILITY, Statistics, COVARIANCE_OF_DECISION_DP
from src.util import load_dictionary
from src.training import _save_results, _process_results
from src.plotting import plot_mean, plot_median

# get imput and output path from args
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_path', type=str, required=True, help="the path containing the raw data")
parser.add_argument('-o', '--output_path', type=str, required=True,
                    help="the path into which the processed data will be saved")
parser.add_argument('-s', '--save', action='store_true')
parser.add_argument('-fsw', '--fairness_sweep', action='store_true')
parser.add_argument('-fdg', '--fairness_dual_gradient', action='store_true')
parser.add_argument('-fs', '--fairness_skip', type=int, required=False,
                    help="only take every #fairness_skip fairness value into account")
parser.add_argument('-fst', '--fairness_start', type=float, required=False)
parser.add_argument('-a', '--analyze', action='store_true')

parser.add_argument('-dp', '--demographic_parity', action='store_true')
parser.add_argument('-eop', '--equality_of_opportunity', action='store_true')
parser.add_argument('-u', '--utility', action='store_true')
parser.add_argument('-acc', '--accuracy', action='store_true')
parser.add_argument('-cov', '--covariance_of_decision', action='store_true')
parser.add_argument('--combine_measures', action='store_true')
args = parser.parse_args()


def combine_runs(runs):
    run_results = []

    # load statistics and model parameters for all runs of a parameter setting
    for run_path, run in runs:
        statistics_path = os.path.join(run_path, "statistics.json")
        serialized_statistics = load_dictionary(statistics_path)
        statistics = Statistics.build_from_serialized_dictionary(serialized_statistics)

        model_parameter_path = os.path.join(run_path, "models.json")
        serialized_model_parameters = load_dictionary(model_parameter_path)
        model_parameters = ModelParameters.build_from_serialized_dictionary(serialized_model_parameters)

        run_results.append((statistics, model_parameters))

    # merge statistics and model parameters
    statistcs_over_runs, model_parameters = _process_results(run_results)

    return statistcs_over_runs, model_parameters


def fairness(lambdas, fairness_skip=None):
    fairness_rates = np.array([float(run) for run_path, run in lambdas], dtype=float)
    lambda_paths = [run_path for run_path, run in lambdas]
    sorted_fairness_idx = np.argsort(fairness_rates)

    if fairness_skip:
        sorted_fairness_idx = np.argsort(fairness_rates)
        sorted_fairness_idx = sorted_fairness_idx[::fairness_skip]

    overall_statistics = MultiStatistics()
    selected_fairness_rates = []

    # load statistics and model parameters for all runs of a parameter setting
    for lambda_idx in sorted_fairness_idx:
        if args.fairness_start is not None and fairness_rates[lambda_idx] < args.fairness_start:
            continue

        selected_fairness_rates.append(fairness_rates[lambda_idx])

        current_lambda_path = lambda_paths[lambda_idx]
        current_lambda_statistics_path = os.path.join(current_lambda_path, "statistics.json")

        runs = [(os.path.join(current_lambda_path, run), run) for run in os.listdir(current_lambda_path)
                if os.path.isdir(os.path.join(current_lambda_path, run))]

        if len(runs) > 0:
            statistics, model_parameters = combine_runs(runs)
            _save_results(current_lambda_path, statistics, model_parameters)
        else:
            serialized_statistics = load_dictionary(current_lambda_statistics_path)
            statistics = Statistics.build_from_serialized_dictionary(serialized_statistics)

        overall_statistics.log_run(statistics)

    return overall_statistics, selected_fairness_rates


def _log_result_row(result_row, statistics, dg):
    utility = statistics.get_additonal_measure(UTILITY, "Utility")
    demographic_parity = statistics.demographic_parity()
    equality_of_opportunity = statistics.equality_of_opportunity()

    if not dg:
        result_row.append(utility.median().max())
        result_row.append(utility.median().min())
    else:
        result_row.append(utility.median()[-1])
        result_row.append(demographic_parity.median()[-1])
        result_row.append(equality_of_opportunity.median()[-1])

    result_row.append((utility.third_quartile() - utility.first_quartile()).mean(axis=0))
    result_row.append((demographic_parity.third_quartile()
                       - demographic_parity.first_quartile()).mean(axis=0))
    result_row.append((equality_of_opportunity.third_quartile()
                       - equality_of_opportunity.first_quartile()).mean(axis=0))


def _save(args, statistics, model_parameters, output_path, x_axis, x_label, x_scale):
    # mkdir output path if necessary
    Path(output_path).mkdir(parents=True, exist_ok=True)
    performance_measures = []
    fairness_measures = []

    if args.utility:
        performance_measures.append(statistics.get_additonal_measure(UTILITY, "Utility"))
    if args.accuracy:
        performance_measures.append(statistics.accuracy())

    if args.demographic_parity:
        fairness_measures.append(statistics.demographic_parity())
    if args.equality_of_opportunity:
        fairness_measures.append(statistics.equality_of_opportunity())
    if args.covariance_of_decision:
        fairness_measures.append(statistics.get_additonal_measure(COVARIANCE_OF_DECISION_DP,
                                                                  "Covariance of decision \n"
                                                                  "(Demographic Parity)"))

    if args.combine_measures:
        performance_measures.extend(fairness_measures)
        fairness_measures = []

    # save the merged statistics, parameters and plot them
    _save_results(output_path, statistics, model_parameters)
    plot_mean(x_values=x_axis,
              x_label=x_label,
              x_scale=x_scale,
              performance_measures=performance_measures,
              fairness_measures=fairness_measures,
              file_path=os.path.join(output_path, "results_mean.png"))
    plot_median(x_values=x_axis,
                x_label=x_label,
                x_scale=x_scale,
                performance_measures=performance_measures,
                fairness_measures=fairness_measures,
                file_path=os.path.join(output_path, "results_median.png"))


cost_directories = [os.path.join(args.input_path, cost) for cost in os.listdir(args.input_path)
                    if os.path.isdir(os.path.join(args.input_path, cost))]

results = []

for cost in cost_directories:
    learning_rates = [(os.path.join(cost, lr), lr) for lr in os.listdir(cost)
                      if os.path.isdir(os.path.join(cost, lr))]

    for learning_rate_path, learning_rate in learning_rates:
        parameter_settings = [(os.path.join(learning_rate_path, parameter_setting), parameter_setting) for
                              parameter_setting in os.listdir(learning_rate_path) if
                              os.path.isdir(os.path.join(learning_rate_path, parameter_setting))]

        for parameter_setting_path, parameter_setting in parameter_settings:
            parameter_settings_subfolders = [(os.path.join(parameter_setting_path, run), run) for run in
                                             os.listdir(parameter_setting_path)
                                             if os.path.isdir(os.path.join(parameter_setting_path, run))]

            if args.analyze:
                if args.fairness_dual_gradient:
                    result_columns = ['Fairness Learning Rate',
                                      'Fairness Epochs',
                                      'Fairness Batch size',
                                      'Final median utility',
                                      'Final median DP',
                                      'Final median EOP',
                                      'Average utility IQR',
                                      'Average DP IQR',
                                      'Average EOP IQR']
                    result_row = []
                else:
                    result_columns = ['Learning Rate',
                                      'Time Steps',
                                      'Epochs',
                                      'Number of Batches',
                                      'Maximum median utility',
                                      'Minimum median utility',
                                      'Average utility IQR',
                                      'Average DP IQR',
                                      'Average EOP IQR']

                    result_row = [re.search("(?<=lr)\d+(\.\d+)*", learning_rate)[0]]
                    result_row.append(re.search("(?<=ts)\d+", parameter_setting)[0])
                    result_row.append(re.search("(?<=ep)\d+", parameter_setting)[0])
                    result_row.append(re.search("(?<=bs)\d+", parameter_setting)[0])
            else:
                result_row = None

            if args.fairness_dual_gradient:
                for fairness_lr_path, fairness_lr in parameter_settings_subfolders:
                    fairness_parameter_settings = [(os.path.join(fairness_lr_path, setting), setting) for setting
                                                   in
                                                   os.listdir(fairness_lr_path)
                                                   if os.path.isdir(os.path.join(fairness_lr_path, setting))]

                    for fairness_settings_path, fairness_settings in fairness_parameter_settings:
                        runs = [(os.path.join(fairness_settings_path, run), run) for run in
                                os.listdir(fairness_settings_path)
                                if os.path.isdir(os.path.join(fairness_settings_path, run))]
                        statistics, model_parameters = combine_runs(runs)
                        utility = statistics.get_additonal_measure(UTILITY, "Utility")
                        x_axis = range(utility.length)
                        x_label = "Time Step"
                        x_scale = "linear"

                        if result_row is not None:
                            result_row = [re.search("(?<=flr)\d+(\.\d+)*", fairness_lr)[0]]
                            result_row.append(re.search("(?<=fe)\d+", fairness_settings)[0])
                            result_row.append(re.search("(?<=fbs)\d+", fairness_settings)[0])
                            _log_result_row(result_row, statistics, True)
                            results.append(result_row)

                        if args.save:
                            output_path = fairness_settings_path.replace(args.input_path, args.output_path)
                            _save(args, statistics, model_parameters, output_path, x_axis, x_label, x_scale)

                        print("finished processing {}".format(fairness_settings_path))
            else:
                if args.fairness_sweep:
                    statistics, x_axis = fairness(parameter_settings_subfolders, args.fairness_skip)
                    model_parameters = None
                    x_label = "Lagrangian Multiplier"
                    x_scale = "log"
                else:
                    statistics, model_parameters = combine_runs(parameter_settings_subfolders)
                    utility = statistics.get_additonal_measure(UTILITY, "Utility")
                    x_axis = range(utility.shape[0])
                    x_label = "Time Step"
                    x_scale = "linear"

                if result_row is not None:
                    _log_result_row(result_row, statistics, False)
                    results.append(result_row)

                if args.save:
                    output_path = parameter_setting_path.replace(args.input_path, args.output_path)
                    _save(args, statistics, model_parameters, output_path, x_axis, x_label, x_scale)

            print("finished processing {}".format(parameter_setting_path))

if args.analyze:
    results_df = pd.DataFrame(results)
    results_df.columns = result_columns
    results_df.to_csv(os.path.join(args.output_path, 'results.csv'))
    results_df.to_excel(os.path.join(args.output_path, 'results.xls'))
