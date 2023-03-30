from fecsep.utils import MarkdownReport, timewindow2str
import os

"""
Use the MarkdownReport class to create output for the gefe_qtree

1. string templates are stored for each evaluation
2. string templates are stored for each forecast
3. report should include
    - plots of catalog
    - plots of forecasts
    - evaluation results
    - metadata from run, (maybe json dump of gefe_qtree class)
"""


def generate_report(experiment, timewindow=-1):
    if isinstance(timewindow, int):
        timewindow = experiment.timewindows[timewindow]
        timestr = timewindow2str(timewindow)
    elif isinstance(timewindow, (list, tuple)):
        timestr = timewindow2str(timewindow)
    else:
        timestr = timewindow
        timewindow = [i for i in experiment.timewindows if
                      timewindow(i) == timestr][0]

    report = MarkdownReport()
    report.add_title(
        f"Testing report {experiment.name}", ''
    )
    report.add_heading("Objectives", level=2)
    objs = [
        "Describe the predictive skills of posited hypothesis about seismogenesis with earthquakes of"
        f" $M>{experiment.magnitudes.min()}$",
    ]
    report.add_list(objs)
    # Generate plot of the catalog

    if experiment.catalog is not None:
        cat_path = experiment._paths[timestr]['catalog']
        figure_path = os.path.splitext(cat_path)[0]
        # relative to top-level directory

        catalog = experiment.catalog
        if experiment.region:
            catalog = catalog.filter_spatial(
                region=experiment.region, in_place=True)

        ax = catalog.plot(plot_args={'basemap': 'stock_img', #todo change
                                     'figsize': (12, 8),
                                     'markersize': 8,
                                     'markercolor': 'black',
                                     'grid_fontsize': 16,
                                     'title': '',
                                     'legend': True
                                     })

        ax.get_figure().tight_layout()
        ax.get_figure().savefig(f"{figure_path}.png")
        report.add_figure(
            f"Testing Catalog",
            figure_path,
            level=2,
            caption="",
            add_ext=True
        )
        from fecsep.utils import magnitude_vs_time

        figure_path = figure_path + '_mt'
        ax = magnitude_vs_time(experiment.catalog)
        ax.get_figure().tight_layout()
        ax.get_figure().savefig(f"{figure_path}.png")

        report.add_figure(
            f"",
            figure_path,
            level=2,
            caption="Evaluation catalog  from "
                    f"{timewindow[0]} until {timewindow[1]}. "  # todo
                    f"Earthquakes are filtered above Mw {experiment.magnitudes.min()}. "
                    "Black circles depict individual earthquakes with its radius proportional to the magnitude.",
            add_ext=True
        )

    report.add_heading(
        "Results",
        level=2,
        text="We apply the following tests to each of the forecasts considered in this experiments. "
             "More information regarding the tests can be found [here](https://docs.cseptesting.org/getting_started/theory.html)."
    )
    test_names = [test.name for test in experiment.tests]
    report.add_list(test_names)

    # Include results from Experiment
    for test in experiment.tests:
        fig_path = experiment._paths[timestr]['figures'][test.name]

        width = test.plot_args.get('figsize', [4])[0] * 96  # inch to pix
        report.add_figure(
            f"{test.name}",
            fig_path,
            level=3,
            caption=test.markdown,
            add_ext=True,
            width=width
        )

    report.table_of_contents()
    report.save(experiment.run_folder)
