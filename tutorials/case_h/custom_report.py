from floatcsep.postprocess.reporting import MarkdownReport
from floatcsep.utils.helpers import timewindow2str


def main(experiment):
    """

    Args:
        experiment: a floatcsep.experiment.Experiment class

    """
    # Access the last time-window
    timewindow = experiment.timewindows[-1]

    # Convert the time-window to a string
    timestr = timewindow2str(timewindow)

    # Instantiates a Report object and adds a title and objectives
    report = MarkdownReport()
    report.add_title(f"Experiment Report - {experiment.name}", "")
    report.add_heading("Objectives", level=2)

    objs = [
        f"Comparison of ETAS, pyMock-Poisson and pyMock-NegativeBinomial models for the"
        f"day after the Amatrice earthquake, for events with M>{min(experiment.magnitudes)}.",
    ]
    report.add_list(objs)

    # Adds an input figure
    report.add_figure(
        f"Input catalog",
        [
            experiment.registry.get_figure_key("main_catalog_map"),
            experiment.registry.get_figure_key("main_catalog_time"),
        ],
        level=3,
        ncols=1,
        caption=f"Evaluation catalog of {experiment.start_date}. "
        f"Earthquakes are filtered above Mw"
        f" {min(experiment.magnitudes)}.",
        add_ext=True,
    )

    # Include results from Experiment
    test = experiment.tests[0]
    for model in experiment.models:
        fig_path = experiment.registry.get_figure_key(timestr, f"{test.name}_{model.name}")
        report.add_figure(
            f"{test.name}: {model.name}",
            fig_path,
            level=3,
            caption="Catalog-based N-test",
            add_ext=True,
            width=200,
        )

    # Stores the report
    report.save(experiment.registry.abs(experiment.registry.run_dir))
