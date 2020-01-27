import matplotlib.pyplot as plt
from src.evaluation import Statistics, LambdaStatistics

def _plot_results(utility, benefit_delta, xaxis, xlable, xscale, utility_uncertainty=None, benefit_delta_uncertainty=None, file_path=None):
    f, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=(25, 10))
    ax1.plot(xaxis, utility)
    ax1.set_xlabel(xlable)
    ax1.set_ylabel("Utility")
    ax1.set_xscale(xscale)
    ax1.fill_between(xaxis, utility_uncertainty[0], utility_uncertainty[1], alpha=0.3, edgecolor='#060080', facecolor='#928CFF')
    ax2.plot(xaxis, benefit_delta)
    ax2.set_xlabel(xlable)
    ax2.set_ylabel("Mean Benefit Delta (Demographic Parity)")
    ax2.set_xscale(xscale)
    ax2.fill_between(xaxis, benefit_delta_uncertainty[0], benefit_delta_uncertainty[1], alpha=0.3, edgecolor='#060080', facecolor='#928CFF')

    if file_path is None:
        plt.show()
    else:
        plt.savefig(file_path)

def plot_median(statistics, file_path=None):
    if isinstance(statistics, LambdaStatistics):
        # Plot median results over range of lambdas:
        _plot_results(
            statistics.performance(measure=Statistics.UTILITY, result_format=Statistics.MEDIAN),
            statistics.fairness(measure=Statistics.DEMOGRAPHIC_PARITY, result_format=Statistics.MEDIAN),
            statistics.results[LambdaStatistics.LAMBDAS],
            "Lambdas",
            "log",
            (statistics.performance(measure=Statistics.UTILITY, result_format=Statistics.FIRST_QUARTILE), statistics.performance(measure=Statistics.UTILITY, result_format=Statistics.THIRD_QUARTILE)),
            (statistics.fairness(measure=Statistics.DEMOGRAPHIC_PARITY, result_format=Statistics.FIRST_QUARTILE), statistics.fairness(measure=Statistics.DEMOGRAPHIC_PARITY, result_format=Statistics.THIRD_QUARTILE)),
            file_path)
    else:
        # Plot median results for single lambda:
        _plot_results(
            statistics.performance(measure=Statistics.UTILITY, result_format=Statistics.MEDIAN),
            statistics.fairness(measure=Statistics.DEMOGRAPHIC_PARITY, result_format=Statistics.MEDIAN),
            range(0, statistics.results[Statistics.TIMESTEPS]),
            "Timestep",
            "linear",
            (statistics.performance(measure=Statistics.UTILITY, result_format=Statistics.FIRST_QUARTILE), statistics.performance(measure=Statistics.UTILITY, result_format=Statistics.THIRD_QUARTILE)),
            (statistics.fairness(measure=Statistics.DEMOGRAPHIC_PARITY, result_format=Statistics.FIRST_QUARTILE), statistics.fairness(measure=Statistics.DEMOGRAPHIC_PARITY, result_format=Statistics.THIRD_QUARTILE)),
            file_path)
        
def plot_mean(statistics, file_path=None):
    u_mean = statistics.performance(measure=Statistics.UTILITY, result_format=Statistics.MEAN)
    u_stddev = statistics.performance(measure=Statistics.UTILITY, result_format=Statistics.STANDARD_DEVIATION)
    bd_mean = statistics.fairness(measure=Statistics.DEMOGRAPHIC_PARITY, result_format=Statistics.MEAN)
    bd_stddev = statistics.fairness(measure=Statistics.DEMOGRAPHIC_PARITY, result_format=Statistics.STANDARD_DEVIATION)

    if isinstance(statistics, LambdaStatistics):
        # Plot median results over range of lambdas:
        _plot_results(
            u_mean,
            bd_mean,
            statistics.results[LambdaStatistics.LAMBDAS],
            "Lambdas",
            "log",
            (u_mean - u_stddev, u_mean + u_stddev),
            (bd_mean - bd_stddev, bd_mean + bd_stddev),
            file_path)
    else:
        # Plot median results for single lambda:
        _plot_results(
            u_mean,
            bd_mean,
            range(0, statistics.results[Statistics.TIMESTEPS]),
            "Timestep",
            "linear",
            (u_mean - u_stddev, u_mean + u_stddev),
            (bd_mean - bd_stddev, bd_mean + bd_stddev),
            file_path)


# def plot_mean_over_lambdas(statistics, file_path=None):
#     """ Plots the mean final utility and benefit delta for a range of different fairness rates.
        
#     Args:
#         statistics: The dicitionary that contains the statistical information that will be plotted.
#         file_path: The path of the file into which the resut plot should be saved. If the file path is
#         None the plot is not saved.
#     """
#     util_mean = statistics.performance(Statistics.UTILITY)[Statistics.MEAN]
#     ben_mean = statistics.performance()
#     _plot_results(
#         util_mean,
#         ben_mean,
#         statistics["lambdas"],
#         "Lambda",
#         "log",
#         (util_mean - statistics["utility"]["stddev"], util_mean + statistics["utility"]["stddev"]),
#         (ben_mean - statistics["benefit_delta"]["stddev"], ben_mean + statistics["benefit_delta"]["stddev"]),
#         file_path)

# def plot_median_over_lambdas(statistics, file_path=None):
#     """ Plots the median final utility and benefit delta for a range of different fairness rates.
        
#     Args:
#         statistics: The dicitionary that contains the statistical information that will be plotted.
#         file_path: The path of the file into which the resut plot should be saved. If the file path is
#         None the plot is not saved.
#     """
#     _plot_results(
#         statistics["utility"]["median"],
#         statistics["benefit_delta"]["median"],
#         statistics["lambdas"],
#         "Lambda",
#         "log",
#         (statistics["utility"]["first_quartile"], statistics["utility"]["third_quartile"]),
#         (statistics["benefit_delta"]["first_quartile"], statistics["benefit_delta"]["third_quartile"]),
#         file_path)


# def plot_median_over_time(statistics, file_path=None):    
#     """ Plots the median utilities and benefit deltas for every time step.
        
#     Args:
#         statistics: The dicitionary that contains the statistical information that will be plotted.
#         file_path: The path of the file into which the resut plot should be saved. If the file path is
#         None the plot is not saved.
#     """
#     _plot_results(
#         statistics["utility"]["median"],
#         statistics["benefit_delta"]["median"],
#         range(0, statistics["utility"]["median"].shape[0]),
#         "Timestep",
#         "linear",
#         (statistics["utility"]["first_quartile"], statistics["utility"]["third_quartile"]),
#         (statistics["benefit_delta"]["first_quartile"], statistics["benefit_delta"]["third_quartile"]),
#         file_path)


# def plot_mean_over_time(statistics, file_path=None):
#     """ Plots the mean utilities and benefit deltas for every time step.
        
#     Args:
#         statistics: The dicitionary that contains the statistical information that will be plotted.
#         file_path: The path of the file into which the resut plot should be saved. If the file path is
#         None the plot is not saved.
#     """
#     util_mean = statistics["utility"]["mean"]
#     ben_mean = statistics["benefit_delta"]["mean"]
#     _plot_results(
#         util_mean,
#         ben_mean,
#         range(0, statistics["utility"]["median"].shape[0]),
#         "Timestep",
#         "linear",
#         (util_mean - statistics["utility"]["stddev"], util_mean + statistics["utility"]["stddev"]),
#         (ben_mean - statistics["benefit_delta"]["stddev"], ben_mean + statistics["benefit_delta"]["stddev"]),
#         file_path)