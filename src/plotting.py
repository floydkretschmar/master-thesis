import os
import sys
root_path = os.path.abspath(os.path.join('.'))
if root_path not in sys.path:
    sys.path.append(root_path)

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from src.training_evaluation import Statistics, ModelParameters

def _plot_results(
        utility, 
        demographic_parity, 
        equality_of_opportunity,
        xaxis, 
        xlable, 
        xscale, 
        utility_uncertainty=None, 
        demographic_parity_uncertainty=None, 
        equality_of_opportunity_uncertainty=None, 
        lambdas=None, 
        lambdas_uncertainty=None, 
        file_path=None):
    if lambdas is not None and lambdas_uncertainty is not None:
        f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharex=True, figsize=(25, 10))
        ax4.plot(xaxis, lambdas)
        ax4.set_xlabel(xlable)
        ax4.set_ylabel("Lambda")
        ax4.set_xscale(xscale)
        #ax4.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax4.fill_between(xaxis, lambdas_uncertainty[0], lambdas_uncertainty[1], alpha=0.3, edgecolor='#060080', facecolor='#928CFF')
    else:
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, figsize=(25, 10))

    ax1.plot(xaxis, utility)
    ax1.set_xlabel(xlable)
    ax1.set_ylabel("Utility")
    ax1.set_xscale(xscale)
    #ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.fill_between(xaxis, utility_uncertainty[0], utility_uncertainty[1], alpha=0.3, edgecolor='#060080', facecolor='#928CFF')
    
    ax2.plot(xaxis, demographic_parity)
    ax2.set_xlabel(xlable)
    ax2.set_ylabel("Benefit Delta (Disparate Impact)")
    ax2.set_xscale(xscale)
    #ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.fill_between(xaxis, demographic_parity_uncertainty[0], demographic_parity_uncertainty[1], alpha=0.3, edgecolor='#060080', facecolor='#928CFF')

    ax3.plot(xaxis, equality_of_opportunity)
    ax3.set_xlabel(xlable)
    ax3.set_ylabel("Benefit Delta (Equality of Opportunity)")
    ax3.set_xscale(xscale)
    #ax3.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax3.fill_between(xaxis, equality_of_opportunity_uncertainty[0], equality_of_opportunity_uncertainty[1], alpha=0.3, edgecolor='#060080', facecolor='#928CFF')

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
        utility=statistics.performance(measure_key=Statistics.UTILITY, result_format=Statistics.MEDIAN),
        demographic_parity=statistics.fairness(measure_key=Statistics.DEMOGRAPHIC_PARITY, result_format=Statistics.MEDIAN),
        equality_of_opportunity=statistics.fairness(measure_key=Statistics.EQUALITY_OF_OPPORTUNITY, result_format=Statistics.MEDIAN),
        xaxis=statistics.results[Statistics.X_VALUES],
        xlable=statistics.results[Statistics.X_NAME],
        xscale=statistics.results[Statistics.X_SCALE],
        utility_uncertainty=
            (statistics.performance(measure_key=Statistics.UTILITY, result_format=Statistics.FIRST_QUARTILE), 
            statistics.performance(measure_key=Statistics.UTILITY, result_format=Statistics.THIRD_QUARTILE)),
        demographic_parity_uncertainty=
            (statistics.fairness(measure_key=Statistics.DEMOGRAPHIC_PARITY, result_format=Statistics.FIRST_QUARTILE), 
            statistics.fairness(measure_key=Statistics.DEMOGRAPHIC_PARITY, result_format=Statistics.THIRD_QUARTILE)),
        equality_of_opportunity_uncertainty=
            (statistics.fairness(measure_key=Statistics.EQUALITY_OF_OPPORTUNITY, result_format=Statistics.FIRST_QUARTILE), 
            statistics.fairness(measure_key=Statistics.EQUALITY_OF_OPPORTUNITY, result_format=Statistics.THIRD_QUARTILE)),
        lambdas=lambdas,
        lambdas_uncertainty=lambdas_uncertainty,
        file_path=file_path)
        
def plot_mean(statistics, file_path=None, model_parameters=None):
    u_mean = statistics.performance(measure_key=Statistics.UTILITY, result_format=Statistics.MEAN)
    u_stddev = statistics.performance(measure_key=Statistics.UTILITY, result_format=Statistics.STANDARD_DEVIATION)

    dp_mean = statistics.fairness(measure_key=Statistics.DEMOGRAPHIC_PARITY, result_format=Statistics.MEAN)
    dp_stddev = statistics.fairness(measure_key=Statistics.DEMOGRAPHIC_PARITY, result_format=Statistics.STANDARD_DEVIATION)

    eop_mean = statistics.fairness(measure_key=Statistics.EQUALITY_OF_OPPORTUNITY, result_format=Statistics.MEAN)
    eop_stddev = statistics.fairness(measure_key=Statistics.EQUALITY_OF_OPPORTUNITY, result_format=Statistics.STANDARD_DEVIATION)

    if model_parameters is not None:
        lambdas = model_parameters.get_lagrangians(result_format=ModelParameters.MEAN)
        lambdas_uncertainty = (model_parameters.get_lagrangians(result_format=ModelParameters.MEAN) - model_parameters.get_lagrangians(result_format=ModelParameters.STANDARD_DEVIATION), 
            model_parameters.get_lagrangians(result_format=ModelParameters.MEAN) + model_parameters.get_lagrangians(result_format=ModelParameters.STANDARD_DEVIATION))
    else:
        lambdas = None
        lambdas_uncertainty = None

    _plot_results(
        utility=u_mean,
        demographic_parity=dp_mean,
        equality_of_opportunity=eop_mean,
        xaxis=statistics.results[Statistics.X_VALUES],
        xlable=statistics.results[Statistics.X_NAME],
        xscale=statistics.results[Statistics.X_SCALE],
        utility_uncertainty=(u_mean - u_stddev, u_mean + u_stddev),
        demographic_parity_uncertainty=(dp_mean - dp_stddev, dp_mean + dp_stddev),
        equality_of_opportunity_uncertainty=(eop_mean - eop_stddev, eop_mean + eop_stddev),
        lambdas=lambdas,
        lambdas_uncertainty=lambdas_uncertainty,
        file_path=file_path)
