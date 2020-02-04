import os
import sys
root_path = os.path.abspath(os.path.join('.'))
if root_path not in sys.path:
    sys.path.append(root_path)

import matplotlib.pyplot as plt
from src.training_evaluation import Statistics, ModelParameters

def _plot_results(utility, benefit_delta, xaxis, xlable, xscale, utility_uncertainty=None, benefit_delta_uncertainty=None, lambdas=None, lambdas_uncertainty=None, file_path=None):
    if lambdas is not None and lambdas_uncertainty is not None:
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, figsize=(25, 10))
        ax3.plot(xaxis, lambdas)
        ax3.set_xlabel(xlable)
        ax3.set_ylabel("Lambda")
        ax3.set_xscale(xscale)
        ax3.fill_between(xaxis, lambdas_uncertainty[0], lambdas_uncertainty[1], alpha=0.3, edgecolor='#060080', facecolor='#928CFF')
    else:
        f, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=(25, 10))

    ax1.plot(xaxis, utility)
    ax1.set_xlabel(xlable)
    ax1.set_ylabel("Utility")
    ax1.set_xscale(xscale)
    ax1.fill_between(xaxis, utility_uncertainty[0], utility_uncertainty[1], alpha=0.3, edgecolor='#060080', facecolor='#928CFF')
    
    ax2.plot(xaxis, benefit_delta)
    ax2.set_xlabel(xlable)
    ax2.set_ylabel("Benefit Delta")
    ax2.set_xscale(xscale)
    ax2.fill_between(xaxis, benefit_delta_uncertainty[0], benefit_delta_uncertainty[1], alpha=0.3, edgecolor='#060080', facecolor='#928CFF')

    if file_path is None:
        plt.show()
    else:
        plt.savefig(file_path)

def plot_median(statistics, file_path=None, model_parameters=None):
    if model_parameters is not None:
        lambdas = model_parameters.get_lagrangians(result_format=ModelParameters.MEDIAN)
        lambdas_uncertainty = (model_parameters.get_lagrangians(result_format=ModelParameters.FIRST_QUARTILE), model_parameters.get_lagrangians(result_format=ModelParameters.THIRD_QUARTILE))
    else:
        lambdas = None
        lambdas_uncertainty = None

    _plot_results(
        utility=statistics.performance(measure=Statistics.UTILITY, result_format=Statistics.MEDIAN),
        benefit_delta=statistics.fairness(measure=Statistics.DEMOGRAPHIC_PARITY, result_format=Statistics.MEDIAN),
        xaxis=statistics.results[Statistics.X_VALUES],
        xlable=statistics.results[Statistics.X_NAME],
        xscale=statistics.results[Statistics.X_SCALE],
        utility_uncertainty=
            (statistics.performance(measure=Statistics.UTILITY, result_format=Statistics.FIRST_QUARTILE), 
            statistics.performance(measure=Statistics.UTILITY, result_format=Statistics.THIRD_QUARTILE)),
        benefit_delta_uncertainty=
            (statistics.fairness(measure=Statistics.DEMOGRAPHIC_PARITY, result_format=Statistics.FIRST_QUARTILE), 
            statistics.fairness(measure=Statistics.DEMOGRAPHIC_PARITY, result_format=Statistics.THIRD_QUARTILE)),
        lambdas=lambdas,
        lambdas_uncertainty=lambdas_uncertainty,
        file_path=file_path)
        
def plot_mean(statistics, file_path=None, model_parameters=None):
    u_mean = statistics.performance(measure=Statistics.UTILITY, result_format=Statistics.MEAN)
    u_stddev = statistics.performance(measure=Statistics.UTILITY, result_format=Statistics.STANDARD_DEVIATION)
    bd_mean = statistics.fairness(measure=Statistics.DEMOGRAPHIC_PARITY, result_format=Statistics.MEAN)
    bd_stddev = statistics.fairness(measure=Statistics.DEMOGRAPHIC_PARITY, result_format=Statistics.STANDARD_DEVIATION)

    if model_parameters is not None:
        lambdas = model_parameters.get_lagrangians(result_format=ModelParameters.MEAN)
        lambdas_uncertainty = (model_parameters.get_lagrangians(result_format=ModelParameters.MEAN) - model_parameters.get_lagrangians(result_format=ModelParameters.STANDARD_DEVIATION), 
            model_parameters.get_lagrangians(result_format=ModelParameters.MEAN) + model_parameters.get_lagrangians(result_format=ModelParameters.STANDARD_DEVIATION))
    else:
        lambdas = None
        lambdas_uncertainty = None

    _plot_results(
        utility=u_mean,
        benefit_delta=bd_mean,
        xaxis=statistics.results[Statistics.X_VALUES],
        xlable=statistics.results[Statistics.X_NAME],
        xscale=statistics.results[Statistics.X_SCALE],
        utility_uncertainty=(u_mean - u_stddev, u_mean + u_stddev),
        benefit_delta_uncertainty=(bd_mean - bd_stddev, bd_mean + bd_stddev),
        lambdas=lambdas,
        lambdas_uncertainty=lambdas_uncertainty,
        file_path=file_path)
