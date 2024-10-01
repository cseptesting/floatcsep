import matplotlib.pyplot as plt
import numpy
from matplotlib import pyplot


def main(experiment):
    """
    Example custom plot function (Observed vs. forecast rates in time)

    Args:
        experiment: a floatcsep.experiment.Experiment class

    """

    # Get all the timewindows
    timewindows = experiment.timewindows

    # Get the pymock model
    model = experiment.get_model("pymock")

    # Initialize the data lists to plot
    window_mid_time = []
    event_counts = []
    rate_mean = []
    rate_2std = []

    for timewindow in timewindows:

        # Get for a given timewindow and the model
        n_test_result = experiment.results_repo.load_results(
            "Catalog_N-test", timewindow, model
        )

        # Append the results
        window_mid_time.append(timewindow[0] + (timewindow[1] - timewindow[0]) / 2)
        event_counts.append(n_test_result.observed_statistic)
        rate_mean.append(numpy.mean(n_test_result.test_distribution))
        rate_2std.append(2 * numpy.std(n_test_result.test_distribution))

    # Create the figure
    fig, ax = plt.subplots(1, 1)

    # Plot the observed number of events vs. time
    ax.plot(window_mid_time, event_counts, "bo", label="Observed catalog")

    # Plot the forecasted mean rate and its error (2 * standard_deviation)
    ax.errorbar(
        window_mid_time,
        rate_mean,
        yerr=rate_2std,
        fmt="o",
        label="PyMock forecast",
        color="red",
    )

    # Format and save figure
    ax.set_xticks([tw[0] for tw in timewindows] + [timewindows[-1][1]])
    fig.autofmt_xdate()
    ax.set_xlabel("Time")
    ax.set_ylabel(r"Number of events $M\geq 3.5$")
    pyplot.legend()
    pyplot.grid()
    pyplot.savefig("results/forecast_events_rates.png")
