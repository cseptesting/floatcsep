from floatcsep.utils import MarkdownReport, timewindow2str

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

    timewindow = experiment.timewindows[timewindow]
    timestr = timewindow2str(timewindow)

    hooks = experiment.report_hook
    report = MarkdownReport()
    report.add_title(
        f"Experiment Report - {experiment.name}", hooks.get('title_text', '')
    )

    report.add_heading("Objectives", level=2)

    objs = [
        "Describe the predictive skills of posited hypothesis about "
        "seismogenesis with earthquakes of"
        f" M>{min(experiment.magnitudes)}.",
    ]

    if hooks.get('objectives', None):
        for i in hooks.get('objectives'):
            objs.append(i)

    report.add_list(objs)

    report.add_heading("Authoritative Data", level=2)

    # Generate catalog plot
    if experiment.catalog is not None:
        experiment.plot_catalog()
        report.add_figure(
            f"Input catalog",
            [experiment.path('catalog_figure'),
             experiment.path('magnitude_time')],
            level=3,
            ncols=1,
            caption="Evaluation catalog from "
                    f"{experiment.start_date} until {experiment.end_date}. "  
                    f"Earthquakes are filtered above Mw"
                    f" {min(experiment.magnitudes)}.",
            add_ext=True
        )

    report.add_heading(
        "Results",
        level=2,
        text="The following tests are applied to each of the experiment's "
             "forecasts. More information regarding the tests can be found "
             "[here]"
             "(https://docs.cseptesting.org/getting_started/theory.html)."
    )

    test_names = [test.name for test in experiment.tests]
    report.add_list(test_names)

    # Include results from Experiment
    for test in experiment.tests:
        fig_path = experiment.path(timestr, 'figures', test)
        width = test.plot_args[0].get('figsize', [4])[0] * 96
        report.add_figure(
            f"{test.name}",
            fig_path,
            level=3,
            caption=test.markdown,
            add_ext=True,
            width=width
        )
        for model in experiment.models:
            try:
                fig_path = experiment.path(timestr, 'figures',
                                               f'{test.name}_{model.name}')
                width = test.plot_args[0].get('figsize', [4])[0] * 96
                report.add_figure(
                    f"{test.name}: {model.name}",
                    fig_path,
                    level=3,
                    caption=test.markdown,
                    add_ext=True,
                    width=width
                )
            except KeyError:
                pass
    report.table_of_contents()
    report.save(experiment.path.abs(experiment.path.rundir))
