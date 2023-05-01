from fecsep.utils import MarkdownReport, timewindow2str, magnitude_vs_time

"""
Use the MarkdownReport class to create output for the experiment 

1. string templates are stored for each evaluation
2. string templates are stored for each forecast
3. report should include
    - plots of catalog
    - plots of forecasts
    - evaluation results
    - metadata from run, (maybe json dump of Experiment class)
"""


def generate_report(experiment, timewindow=-1):

    if isinstance(timewindow, (int, float)):
        timewindow = experiment.timewindows[timewindow]
        timestr = timewindow2str(timewindow)
    elif isinstance(timewindow, (list, tuple)):
        timestr = timewindow2str(timewindow)
    else:
        timestr = timewindow
        timewindow = [i for i in experiment.timewindows if
                      timewindow[i] == timestr][0]

    report = MarkdownReport()
    report.add_title(
        f"Experiment Report - {experiment.name}", ''
    )
    report.add_heading("Objectives", level=2)
    objs = [
        "Describe the predictive skills of posited hypothesis about "
        "seismogenesis with earthquakes of"
        f" M>{min(experiment.magnitudes)}",
    ]
    report.add_list(objs)

    # Generate catalog plot
    if experiment.catalog is not None:
        experiment.plot_catalog()
        report.add_figure(
            f"",
            [experiment.tree(timestr, 'figures', 'catalog'),
             experiment.tree(timestr, 'figures', 'magnitude_time')],
            level=2,
            caption="Evaluation catalog  from "
                    f"{timewindow[0]} until {timewindow[1]}. "  
                    f"Earthquakes are filtered above Mw"
                    f" {min(experiment.magnitudes)}.",
            add_ext=True
        )

    report.add_heading(
        "Results",
        level=2,
        text="The following tests are applied to each of the experiment's "
             "forecasts. More information regarding the "
             "tests can be found [here]"
             "(https://docs.cseptesting.org/getting_started/theory.html)."
    )
    test_names = [test.name for test in experiment.tests]
    report.add_list(test_names)

    # Include results from Experiment
    for test in experiment.tests:
        fig_path = experiment.tree(timestr, 'figures', test)
        width = test.plot_args.get('figsize', [4])[0] * 96  #
        report.add_figure(
            f"{test.name}",
            fig_path,
            level=3,
            caption=test.markdown,
            add_ext=True,
            width=width
        )

    report.table_of_contents()
    report.save(experiment.tree.run_folder)
