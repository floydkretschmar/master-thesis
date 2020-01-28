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
