import importlib.util
import itertools
import logging
import os
from typing import TYPE_CHECKING

import numpy

from floatcsep.experiment import ExperimentComparison
from floatcsep.utils.helpers import timewindow2str, str2timewindow

if TYPE_CHECKING:
    from floatcsep.experiment import Experiment


log = logging.getLogger("floatLogger")

"""
Use the MarkdownReport class to create output for the experiment.

1. string templates are stored for each evaluation
2. string templates are stored for each forecast
3. report should include
    - plots of catalog
    - plots of forecasts
    - evaluation results
    - metadata from run, (maybe json dump of Experiment class)
"""


def generate_report(experiment, timewindow=-1):

    report_function = experiment.postprocess.get("report")
    if report_function:
        custom_report(report_function, experiment)
        return

    timewindow = experiment.timewindows[timewindow]
    timestr = timewindow2str(timewindow)

    log.info(f"Saving report into {experiment.registry.run_dir}")

    report = MarkdownReport()
    report.add_title(f"Experiment Report - {experiment.name}", "")
    report.add_heading("Objectives", level=2)

    objs = [
        "Describe the predictive skills of posited hypothesis about "
        "seismogenesis with earthquakes of"
        f" M>{min(experiment.magnitudes)}.",
    ]

    report.add_list(objs)

    report.add_heading("Authoritative Data", level=2)

    # Generate catalog plot
    if experiment.catalog_repo.catalog is not None:
        report.add_figure(
            "Input catalog",
            [
                experiment.registry.get_figure("main_catalog_map"),
                experiment.registry.get_figure("main_catalog_time"),
            ],
            level=3,
            ncols=1,
            caption="Evaluation catalog from "
            f"{experiment.start_date} until {experiment.end_date}. "
            f"Earthquakes are filtered above Mw"
            f" {min(experiment.magnitudes)}.",
            add_ext=True,
        )
    test_names = [test.name for test in experiment.tests]
    report.add_list(test_names)

    report.add_heading("Test results", level=2)

    # Include results from Experiment
    for test in experiment.tests:
        fig_path = experiment.registry.get_figure(timestr, test)
        width = test.plot_args[0].get("figsize", [4])[0] * 96
        report.add_figure(
            f"{test.name}", fig_path, level=3, caption=test.markdown, add_ext=True, width=width
        )
        for model in experiment.models:
            try:
                fig_path = experiment.registry.get_figure(timestr, f"{test.name}_{model.name}")
                width = test.plot_args[0].get("figsize", [4])[0] * 96
                report.add_figure(
                    f"{test.name}: {model.name}",
                    fig_path,
                    level=3,
                    caption=test.markdown,
                    add_ext=True,
                    width=width,
                )
            except KeyError:
                pass
    report.table_of_contents()
    report.save(experiment.registry.abs(experiment.registry.run_dir))


def reproducibility_report(exp_comparison: "ExperimentComparison"):

    numerical = exp_comparison.num_results
    data = exp_comparison.file_comp
    outname = os.path.join("reproducibility_report.md")
    save_path = os.path.dirname(
        os.path.join(
            exp_comparison.reproduced.registry.workdir,
            exp_comparison.reproduced.registry.run_dir,
        )
    )
    report = MarkdownReport(out_name=outname)
    report.add_title(f"Reproducibility Report - {exp_comparison.original.name}", "")

    report.add_heading("Objectives", level=2)
    objs = [
        "Analyze the statistic reproducibility and data reproducibility of"
        " the experiment. Compares the differences between "
        "(i) the original and reproduced scores,"
        " (ii) the statistical descriptors of the test distributions,"
        " (iii) The p-value of a Kolmogorov-Smirnov test -"
        " values beneath 0.1 means we can't reject the distributions are"
        " similar -,"
        " (iv) Hash (SHA-256) comparison between the results' files and "
        "(v) byte-to-byte comparison"
    ]

    report.add_list(objs)
    for num, dat in zip(numerical.items(), data.items()):

        res_keys = list(num[1].keys())
        is_time = False
        try:
            str2timewindow(res_keys[0])
            is_time = True
        except ValueError:
            pass
        if is_time:
            report.add_heading(num[0], level=2)
            for tw in res_keys:
                rows = [
                    [
                        tw,
                        "Score difference",
                        "Test Mean  diff.",
                        "Test Std  diff.",
                        "Test Skew  diff.",
                        "KS-test p value",
                        "Hash (SHA-256) equal",
                        "Byte-to-byte equal",
                    ]
                ]

                for model_stat, model_file in zip(num[1][tw].items(), dat[1][tw].items()):
                    obs = model_stat[1]["observed_statistic"]
                    test = model_stat[1]["test_statistic"]
                    rows.append(
                        [
                            model_stat[0],
                            obs,
                            *[f"{i:.1e}" for i in test[:-1]],
                            f"{test[-1]:.1e}",
                            model_file[1]["hash"],
                            model_file[1]["byte2byte"],
                        ]
                    )
                report.add_table(rows)
        else:
            report.add_heading(num[0], level=2)
            rows = [
                [
                    res_keys[-1],
                    "Max Score difference",
                    "Hash (SHA-256) equal",
                    "Byte-to-byte equal",
                ]
            ]

            for model_stat, model_file in zip(num[1].items(), dat[1].items()):
                obs = numpy.nanmax(model_stat[1]["observed_statistic"])

                rows.append(
                    [
                        model_stat[0],
                        f"{obs:.1e}",
                        model_file[1]["hash"],
                        model_file[1]["byte2byte"],
                    ]
                )

            report.add_table(rows)
    report.table_of_contents()
    report.save(save_path)


def custom_report(report_function: str, experiment: "Experiment"):

    try:
        script_path, func_name = report_function.split(".py:")
        script_path += ".py"

    except ValueError:
        log.error(
            f"Invalid format for custom plot function: {report_function}. "
            "Try {script_name}.py:{func}"
        )
        log.info(
            "\t Skipping reporting. The configuration script can be modified and re-run the"
            " reporting (and plots) only by typing 'floatcsep plot {config}'"
        )
        return

    log.info(f"Creating report from script {script_path} and function {func_name}")
    script_abs_path = experiment.registry.abs(script_path)
    allowed_directory = os.path.dirname(experiment.registry.abs(experiment.config_file))

    if not os.path.isfile(script_path) or (
        os.path.dirname(script_abs_path) != os.path.realpath(allowed_directory)
    ):

        log.error(f"Script {script_path} is not in the configuration file directory.")
        log.info(
            "\t Skipping reporting. The script can be reallocated and re-run the reporting only"
            " by typing 'floatcsep plot {config}'"
        )
        return

    module_name = os.path.splitext(os.path.basename(script_abs_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, script_abs_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    try:
        func = getattr(module, func_name)

    except AttributeError:
        log.error(f"Function {func_name} not found in {script_path}")
        log.info(
            "\t Skipping reporting. Report script can be modified and re-run the report only"
            " by typing 'floatcsep plot {config}'"
        )
        return

    try:
        func(experiment)
    except Exception as e:
        log.error(f"Error executing {func_name} from {script_path}: {e}")
        log.info(
            "\t Skipping reporting. Report script can be modified and re-run the report only"
            " by typing 'floatcsep plot {config}'"
        )
    return


class MarkdownReport:
    """Class to generate a Markdown report from a study."""

    def __init__(self, out_name="report.md"):
        self.out_name = out_name
        self.toc = []
        self.has_title = True
        self.has_introduction = False
        self.markdown = []

    def add_introduction(self, adict):
        """Generate document header from dictionary."""
        first = (
            f"# CSEP Testing Results: {adict['simulation_name']}  \n"
            f"**Forecast Name:** {adict['forecast_name']}  \n"
            f"**Simulation Start Time:** {adict['origin_time']}  \n"
            f"**Evaluation Time:** {adict['evaluation_time']}  \n"
            f"**Catalog Source:** {adict['catalog_source']}  \n"
            f"**Number Simulations:** {adict['num_simulations']}\n"
        )

        # used to determine to place TOC at beginning of document or after
        # introduction.

        self.has_introduction = True
        self.markdown.append(first)
        return first

    def add_text(self, text):
        """
        Text should be a list of strings where each string will be on its own.

        line. Each add_text command represents a paragraph.

        Args:
            text (list): lines to write
        Returns:
        """
        self.markdown.append("  ".join(text) + "\n\n")

    def add_figure(
        self,
        title,
        relative_filepaths,
        level=2,
        ncols=1,
        add_ext=False,
        text="",
        caption="",
        width=None,
    ):
        """
        This function expects a list of filepaths.

        If you want the output
        stacked, select a value of ncols. ncols should be divisible by
        filepaths.

        Args:
            width:
            caption:
            text:
            add_ext:
            ncols:
            title: name of the figure
            level (int): value 1-6 depending on the heading
            relative_filepaths (str or List[Tuple[str]]): list of paths in
                order to make table
        Returns:
        """
        # verify filepaths have proper extension should always be png
        is_single = False
        paths = []
        if isinstance(relative_filepaths, str):
            is_single = True
            paths.append(relative_filepaths)
        else:
            paths = relative_filepaths

        # make "relative path" (to experiment dir) relative to report
        paths = [p.replace("results/", "") for p in paths]

        correct_paths = []
        if add_ext:
            for fp in paths:
                correct_paths.append(fp + ".png")
        else:
            correct_paths = paths

        # generate new lists with size ncols
        formatted_paths = [correct_paths[i : i + ncols] for i in range(0, len(paths), ncols)]

        # convert str into a list, where each potential row is an iter not str
        def build_header(_row):
            top = "|"
            bottom = "|"
            for i, _ in enumerate(_row):
                if i == ncols:
                    break
                top += " |"
                bottom += " --- |"
            return top + "\n" + bottom

        size_ = bool(width) * f"width={width}"

        def add_to_row(_row):
            if len(_row) == 1:
                return f'<img src="{_row[0]}" {size_}/>'
            string = "| "
            for item in _row:
                string = string + f'<img src="{item}" width={width}/>'
            return string

        level_string = f"{level * '#'}"
        result_cell = []
        locator = title.lower().replace(" ", "_")
        result_cell.append(f'{level_string} {title}  <a name="{locator}"></a>\n')
        result_cell.append(f"{text}\n")

        for i, row in enumerate(formatted_paths):
            if i == 0 and not is_single and ncols > 1:
                result_cell.append(build_header(row))
            result_cell.append(add_to_row(row))
        result_cell.append("\n")
        result_cell.append(f"{caption}")

        self.markdown.append("\n".join(result_cell) + "\n")

        # generate metadata for TOC
        self.toc.append((title, level, locator))

    def add_heading(self, title, level=1, text="", add_toc=True):
        # multiplying char simply repeats it
        if isinstance(text, str):
            text = [text]
        cell = []
        level_string = f"{level * '#'}"
        locator = title.lower().replace(" ", "_")
        sub_heading = f'{level_string} {title} <a name="{locator}"></a>\n'
        cell.append(sub_heading)
        try:
            for item in list(text):
                cell.append(item)
        except Exception as ex:
            raise RuntimeWarning(f"Unable to add document subhead, text must be iterable. {ex}")
        self.markdown.append("\n".join(cell) + "\n")

        # generate metadata for TOC
        if add_toc:
            self.toc.append((title, level, locator))

    def add_list(self, _list):
        cell = []
        for item in _list:
            cell.append(f"* {item}")
        self.markdown.append("\n".join(cell) + "\n\n")

    def add_title(self, title, text):
        self.has_title = True
        self.add_heading(title, 1, text, add_toc=False)

    def table_of_contents(self):
        """Generates table of contents based on contents of document."""
        if len(self.toc) == 0:
            return
        toc = ["# Table of Contents"]

        for i, elem in enumerate(self.toc):
            title, level, locator = elem
            space = "   " * (level - 1)
            toc.append(f"{space}1. [{title}](#{locator})")
        insert_loc = 1 if self.has_title else 0
        self.markdown.insert(insert_loc, "\n".join(toc) + "\n\n")

    def add_table(self, data, use_header=True):
        """
        Generates table from HTML and styles using bootstrap class.

        Args:
           data List[Tuple[str]]: should be (nrows, ncols) in size. all rows
            should be the same sizes
        Returns:
            table (str): this can be added to subheading or other cell if
                desired.
        """
        table = ['<div class="table table-striped">', "<table>"]

        def make_header(row_):
            header = ["<tr>"]
            for item in row_:
                header.append(f"<th>{item}</th>")
            header.append("</tr>")
            return "\n".join(header)

        def add_row(row_):
            table_row = ["<tr>"]
            for item in row_:
                table_row.append(f"<td>{item}</td>")
            table_row.append("</tr>")
            return "\n".join(table_row)

        for i, row in enumerate(data):
            if i == 0 and use_header:
                table.append(make_header(row))
            else:
                table.append(add_row(row))
        table.append("</table>")
        table.append("</div>")
        table = "\n".join(table)
        self.markdown.append(table + "\n\n")

    def save(self, save_dir):
        output = list(itertools.chain.from_iterable(self.markdown))
        full_md_fname = os.path.join(save_dir, self.out_name)
        with open(full_md_fname, "w") as f:
            f.writelines(output)
