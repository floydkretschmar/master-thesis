import matplotlib.pyplot as plt

def plot_results_over_time(utility, benefit_delta):
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

#def plot_results_over_lambdas(utility_mean, utility_max, utility_min, benefit_delta_mean, benefit_delta_max, benefit_delta_min, lambdas, file_path=None):
def plot_results_over_lambdas(utility_mean, utility_stddev, benefit_delta_mean, benefit_stddev, lambdas, file_path=None):
    f, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=(25, 10))
    ax1.plot(lambdas, utility_mean)
    ax1.set_xlabel("Lambda")
    ax1.set_ylabel("Utility")
    ax1.set_xscale('log')
    ax1.fill_between(lambdas, utility_mean-utility_stddev, utility_mean+utility_stddev, alpha=0.3, edgecolor='#060080', facecolor='#928CFF')
    ax2.plot(lambdas, benefit_delta_mean)
    ax2.set_xlabel("Lambda")
    ax2.set_ylabel("Mean Benefit Delta")
    ax2.set_xscale('log')
    ax2.fill_between(lambdas, benefit_delta_mean-benefit_stddev, benefit_delta_mean+benefit_stddev, alpha=0.3, edgecolor='#060080', facecolor='#928CFF')

    if file_path is None:
        plt.show()
    else:
        plt.savefig(file_path)