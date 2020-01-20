import matplotlib.pyplot as plt

def _plot_results_over_lambdas(utility, benefit_delta, lambdas, utility_uncertainty=None, benefit_delta_uncertainty=None, file_path=None):
    f, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=(25, 10))
    ax1.plot(lambdas, utility)
    ax1.set_xlabel("Lambda")
    ax1.set_ylabel("Utility")
    ax1.set_xscale('log')
    ax1.fill_between(lambdas, utility_uncertainty[0], utility_uncertainty[1], alpha=0.3, edgecolor='#060080', facecolor='#928CFF')
    ax2.plot(lambdas, benefit_delta)
    ax2.set_xlabel("Lambda")
    ax2.set_ylabel("Mean Benefit Delta")
    ax2.set_xscale('log')
    ax2.fill_between(lambdas, benefit_delta_uncertainty[0], benefit_delta_uncertainty[1], alpha=0.3, edgecolor='#060080', facecolor='#928CFF')

    if file_path is None:
        plt.show()
    else:
        plt.savefig(file_path)

def plot_mean_over_lambdas(statistics, file_path=None):
    """ Plots the mean final utility and benefit delta for a range of different fairness rates.
        
    Args:
        statistics: The dicitionary that contains the statistical information that will be plotted.
        file_path: The path of the file into which the resut plot should be saved. If the file path is
        None the plot is not saved.
    """
    util_mean = statistics["utility"]["mean"]
    ben_mean = statistics["benefit_delta"]["mean"]
    _plot_results_over_lambdas(
        util_mean,
        ben_mean,
        statistics["lambdas"],
        (util_mean - statistics["utility"]["stddev"], util_mean + statistics["utility"]["stddev"]),
        (ben_mean - statistics["benefit_delta"]["stddev"], ben_mean + statistics["benefit_delta"]["stddev"]),
        file_path)

def plot_median_over_lambdas(statistics, file_path=None):
    """ Plots the median final utility and benefit delta for a range of different fairness rates.
        
    Args:
        statistics: The dicitionary that contains the statistical information that will be plotted.
        file_path: The path of the file into which the resut plot should be saved. If the file path is
        None the plot is not saved.
    """
    _plot_results_over_lambdas(
        statistics["utility"]["median"],
        statistics["benefit_delta"]["median"],
        statistics["lambdas"],
        (statistics["utility"]["first_quartile"], statistics["utility"]["third_quartile"]),
        (statistics["benefit_delta"]["first_quartile"], statistics["benefit_delta"]["third_quartile"]),
        file_path)

def plot_results_over_time(utility, benefit_delta):    
    """ Plots the results of one particular training run over all time steps.
        
    Args:
        utility: The utility of the policy at every time step t.
        benefit_delta: The benefit delta of the policy at every time step t.
    """
    utility_mean = utility.mean(axis=0)
    benefit_delta_mean = benefit_delta.mean(axis=0)
    time_steps = range(1, utility.shape[1] + 1)

    f, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=(25,10))
    ax1.plot(time_steps, utility_mean)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Mean Utility")
    ax2.plot(time_steps, benefit_delta_mean)
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Mean Benefit Delta")
    plt.show()