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