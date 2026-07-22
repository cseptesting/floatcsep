import importlib.util
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import numpy
from PIL import Image
from markdown_it import MarkdownIt
from weasyprint import HTML

from floatcsep.experiment import ExperimentComparison
from floatcsep.postprocess import plot_handler
from floatcsep.utils.helpers import str2timewindow, timewindow2str

if TYPE_CHECKING:
    from floatcsep.experiment import Experiment

log = logging.getLogger("floatLogger")

"""
Use the MarkdownReport class to create output for the experiment.

Report includes:
    - plots of catalogs
    - plots of forecasts
    - evaluation results
    - metadata from the run
"""

_MD = MarkdownIt("commonmark", {"html": True}).enable("table")
BASE_TEXT_WIDTH_PX = 800

LOGO_PATH = Path(__file__).resolve().parent / "artifacts" / "logo.png"
FONT_REGULAR_PATH = Path(__file__).resolve().parent / "artifacts" / "NotoSans-Regular.ttf"
FONT_BOLD_PATH = Path(__file__).resolve().parent / "artifacts" / "NotoSans-Bold.ttf"


# ---------------------------------------------------------------------------
# Main reports
# ---------------------------------------------------------------------------
def generate_report(experiment: "Experiment", timewindow: int = -1) -> None:
    """Create the main experiment report (Markdown + PDF)."""

    # postprocess.report:
    #   - str  -> custom report "script.py:func"
    #   - dict -> layout config {catalog_width, forecast_width, results_width}
    #   - None -> default behaviour
    report_cfg = experiment.postprocess.get("report")

    if isinstance(report_cfg, str):
        custom_report(report_cfg, experiment)
        return

    report_cfg = report_cfg if isinstance(report_cfg, dict) else {}

    report_path = experiment.registry.run_dir / "report.md"
    report_dir = report_path.parent

    catalog_width = report_cfg.get("catalog_width", None)
    forecast_width = report_cfg.get("forecast_width", None)
    results_width = report_cfg.get("results_width", None)

    all_windows = list(experiment.time_windows)
    if timewindow == 0:
        windows = all_windows
    else:
        windows = [all_windows[timewindow]]
    show_tw_heading = len(all_windows) > 1

    log.info(f"Saving Markdown report into {report_path}")

    report = MarkdownReport(root_dir=report_dir)
    report.add_title("Experiment Report", experiment.name)
    report.add_text(
        [
            "This experiment evaluates the performance of earthquake forecast models "
            "within a fully specified and reproducible testing framework. This report "
            "summarizes the main results."
        ]
    )

    model_names = ", ".join(m.name for m in experiment.models)
    test_names = ", ".join(t.name for t in experiment.tests)
    meta = {
        "Start date": str(experiment.start_date),
        "End date": str(experiment.end_date),
        "Class": (
            "Time-Dependent"
            if experiment.exp_class in ("td", "time-dependent")
            else "Time-Independent"
        ),
        "Magnitude range": f"{experiment.mag_min} ≤ Mw ≤ {experiment.mag_max}",
        "Region": getattr(experiment.region, "name", str(experiment.region)),
        "Catalog": getattr(experiment.catalog_repo, "name", "unknown"),
        "Models": model_names,
        "Evaluations": test_names,
    }
    report.add_heading("Experiment metadata", level=2)
    report.add_introduction(meta)

    report.add_heading("Objectives", level=2)
    report.add_list(
        [
            "Ensure transparent and reproducible evaluation of submitted models.",
            "Compare forecasts against authoritative seismicity observations.",
        ]
    )

    # Authoritative data / catalog plots
    plot_catalog: dict = plot_handler.parse_plot_config(
        experiment.postprocess.get("plot_catalog", {})
    )
    if experiment.catalog_repo.catalog is not None and isinstance(plot_catalog, dict):
        report.add_heading("Authoritative Data", level=2)
        cat_map = experiment.registry.get_figure_key("main_catalog_map")
        cat_time = experiment.registry.get_figure_key("main_catalog_time")

        report.add_figure(
            "Input catalog",
            [cat_map, cat_time],
            level=3,
            ncols=1,
            caption=(
                "Evaluation catalog from "
                f"{experiment.start_date} until {experiment.end_date}. "
                f"Earthquakes are filtered above Mw {min(experiment.magnitudes)}."
            ),
            width=catalog_width,
        )

    # Forecast plots
    plot_forecasts: dict = plot_handler.parse_plot_config(
        experiment.postprocess.get("plot_forecasts", {})
    )
    if isinstance(plot_forecasts, dict):
        report.add_heading("Forecasts", level=2)
        for tw in windows:
            tw_str = timewindow2str(tw)
            if show_tw_heading:
                report.add_heading(f"Forecasts for {tw_str}", level=3)
            model_level = 4 if show_tw_heading else 3

            for model in experiment.models:
                forecast = experiment.registry.get_figure_key(tw_str, "forecasts", model.name)
                report.add_figure(
                    title=f"{model.name}",
                    fig_path=forecast,
                    level=model_level,
                    width=forecast_width,
                )

    # Test result plots
    report.add_heading("Test results", level=2)
    for tw in windows:
        tw_str = timewindow2str(tw)
        if show_tw_heading:
            report.add_heading(f"Results for {tw_str}", level=3)
        test_level = 4 if show_tw_heading else 3
        model_level = test_level + 1

        for test in experiment.tests:
            result = experiment.registry.get_figure_key(tw_str, test)
            report.add_figure(
                f"{test.name}",
                result,
                level=test_level,
                caption=test.markdown,
                width=results_width,
            )

            if "per_model" not in getattr(test, "plot_modes", []):
                continue

            for model in experiment.models:
                try:
                    result = experiment.registry.get_figure_key(
                        tw_str, f"{test.name}_{model.name}"
                    )
                except KeyError:
                    continue
                if not os.path.isfile(result):
                    continue
                report.add_figure(
                    f"{model.name}",
                    result,
                    level=model_level,
                    caption=test.markdown,
                    width=results_width,
                )

    report.table_of_contents()
    report.save(report_path)

    pdf_path = report_path.with_suffix(".pdf")
    log.info(f"Saving PDF report into {pdf_path}")
    markdown_to_pdf(
        markdown_source=report_path,
        pdf_path=pdf_path,
        base_url=report_dir,
    )


def reproducibility_report(exp_comparison: "ExperimentComparison") -> None:
    """Create the reproducibility report in Markdown."""
    numerical = exp_comparison.num_results
    data = exp_comparison.file_comp

    report_path = (
        exp_comparison.reproduced.registry.workdir
        / exp_comparison.reproduced.registry.run_dir
        / "reproducibility_report.md"
    )

    report = MarkdownReport(root_dir=report_path.parent)
    report.add_title("Reproducibility Report", exp_comparison.original.name)

    report.add_heading("Objectives", level=2)
    objs = [
        "Analyze the statistic reproducibility and data reproducibility of "
        "the experiment. Compares the differences between "
        "(i) the original and reproduced scores, "
        "(ii) the statistical descriptors of the test distributions, "
        "(iii) the p-value of a Kolmogorov-Smirnov test "
        "(values below 0.1 mean we cannot reject that the "
        "distributions are similar), "
        "(iv) hash (SHA-256) comparison between the result files and "
        "(v) byte-to-byte comparison.",
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
                        "Test mean diff.",
                        "Test std diff.",
                        "Test skew diff.",
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
                    "Max score difference",
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
    report.save(report_path)


def custom_report(report_function: str, experiment: "Experiment") -> None:
    """Run a user-provided report function from a script."""
    try:
        script_path, func_name = report_function.split(".py:")
        script_path += ".py"
    except ValueError:
        log.error(
            f"Invalid format for custom report function: {report_function}. "
            "Try {script_name}.py:{func}"
        )
        log.info(
            "\tSkipping reporting. The configuration script can be modified "
            "and the reporting re-run with 'floatcsep plot {config}'."
        )
        return

    log.info(f"Creating report from script {script_path} and function {func_name}")
    script_abs_path = experiment.registry.abs(script_path)
    allowed_directory = os.path.dirname(experiment.registry.abs(experiment.config_file))

    if not os.path.isfile(script_path) or (
        os.path.dirname(script_abs_path) != os.path.realpath(allowed_directory)
    ):
        log.error(f"Script {script_path} is not in the configuration directory.")
        log.info(
            "\tSkipping reporting. The script can be reallocated and "
            "reporting re-run with 'floatcsep plot {config}'."
        )
        return

    module_name = os.path.splitext(os.path.basename(script_abs_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, script_abs_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]

    try:
        func = getattr(module, func_name)
    except AttributeError:
        log.error(f"Function {func_name} not found in {script_path}")
        log.info(
            "\tSkipping reporting. Report script can be modified and "
            "re-run with 'floatcsep plot {config}'."
        )
        return

    try:
        func(experiment)
    except Exception as exc:  # pragma: no cover - user code
        log.error(f"Error executing {func_name} from {script_path}: {exc}")
        log.info(
            "\tSkipping reporting. Report script can be modified and "
            "re-run with 'floatcsep plot {config}'."
        )


# ---------------------------------------------------------------------------
# Markdown builder
# ---------------------------------------------------------------------------
class MarkdownReport:
    """Helper class to build a Markdown report."""

    def __init__(self, root_dir: Union[str, Path]) -> None:
        self.root_dir = Path(root_dir)
        self.toc = []
        self.has_title = True
        self.has_introduction = False
        self.markdown = []

    def add_title(self, title, subtitle: str = "") -> None:
        """
        Add the main report title.

        Layout:
            - Left: experiment name (subtitle) as main H1,
                    title as a slightly smaller line below.
            - Right: floatCSEP logo, a bit larger.
        """
        self.has_title = True

        main_text = title or subtitle
        secondary_text = subtitle if subtitle else ""
        locator = main_text.lower().replace(" ", "_")

        logo_path = os.path.relpath(LOGO_PATH, self.root_dir)

        html = (
            "<div style='overflow:auto;'>\n"
            f"  <img src='{logo_path}' class='figure-img' "
            "style='float:right; margin-left:1em; width:130px; height:auto;' />\n"
            f"  <h1 style='margin:0;'><a id='{locator}'></a>{main_text}</h1>\n"
            f"  <p style='margin:0; font-size:1.6em;'>{secondary_text}</p>\n"
            "</div>\n\n"
        )
        self.markdown.append(html)

    def add_introduction(self, meta: dict) -> str:
        """
        Add an experiment metadata block from a dictionary.

        Expects a mapping like:
            {"Start date": "...", "End date": "...", ...}
        """
        lines = []
        for key, value in meta.items():
            lines.append(f"- **{key}:** {value}")
        block = "\n".join(lines) + "\n\n"

        self.has_introduction = True
        self.markdown.append(block)
        return block

    def add_text(self, text) -> None:
        """Add a paragraph from a list of lines."""
        self.markdown.append("  ".join(text) + "\n\n")

    def add_figure(
        self,
        title,
        fig_path,
        level: int = 2,
        ncols: int = 1,
        text: str = "",
        caption: str = "",
        width: Optional[float] = None,
    ) -> None:
        """
        Add one or more figures to the report.

        'width' is a fraction of the text width (0 < width <= 1). If None,
        a default is chosen automatically based on aspect ratio and ncols.
        """
        is_single = False
        paths = []
        if isinstance(fig_path, str):
            is_single = True
            paths.append(os.path.relpath(fig_path, self.root_dir))
        else:
            paths = [os.path.relpath(i, self.root_dir) for i in fig_path]

        formatted_paths = [paths[i : i + ncols] for i in range(0, len(paths), ncols)]

        if width is not None:
            frac = max(0.1, min(1.0, float(width)))
        else:
            if paths:
                abs_path = self.root_dir / paths[0]
                aspect = get_image_aspect(abs_path)
            else:
                aspect = None
            frac = width_fraction_from_aspect(aspect, ncols=ncols)

        pct = int(round(frac * 100.0))
        px = int(round(frac * BASE_TEXT_WIDTH_PX))

        def build_header(ncols_):
            header = "| " + " | ".join([" "] * ncols_) + " |"
            under = "| " + " | ".join(["---"] * ncols_) + " |"
            return header + "\n" + under

        def img_tag(path_):
            style = (
                f"display:block; margin:0.5em auto; "
                f"width:{pct}%; max-width:100%; height:auto;"
            )
            size_attr = f' width="{px}"'
            return f'<img src="{path_}" class="figure-img" ' f'style="{style}"{size_attr}/>'

        def add_to_row(row_):
            if len(row_) == 1:
                return img_tag(row_[0])
            cells = [img_tag(item) for item in row_]
            return "| " + " | ".join(cells) + " |"

        level_string = f"{level * '#'}"
        result_cell = []
        locator = title.lower().replace(" ", "_")
        result_cell.append(f'{level_string} {title}  <a id="{locator}"></a>\n')
        result_cell.append(f"{text}\n")

        for i, row in enumerate(formatted_paths):
            if i == 0 and not is_single and ncols > 1:
                result_cell.append(build_header(len(row)))
            result_cell.append(add_to_row(row))

        result_cell.append("\n")
        result_cell.append(f"{caption}")

        self.markdown.append("\n".join(result_cell) + "\n")
        self.toc.append((title, level, locator))

    def add_heading(
        self,
        title,
        level: int = 1,
        text: str = "",
        add_toc: bool = True,
    ) -> None:
        """Add a heading with optional text and TOC entry."""
        if isinstance(text, str):
            text = [text]
        cell = []
        level_string = f"{level * '#'}"
        locator = title.lower().replace(" ", "_")
        sub_heading = f'{level_string} {title} <a id="{locator}"></a>\n'
        cell.append(sub_heading)
        for item in list(text):
            cell.append(item)
        self.markdown.append("\n".join(cell) + "\n")

        if add_toc:
            self.toc.append((title, level, locator))

    def add_list(self, items) -> None:
        """Add a bulleted list."""
        cell = [f"* {item}" for item in items]
        self.markdown.append("\n".join(cell) + "\n\n")

    def table_of_contents(self) -> None:
        """Generate a Table of Contents from top-level headings (H2)."""
        if not self.toc:
            return

        max_level = 2  # include only headings with level <= 2
        entries = [
            (title, level, locator) for title, level, locator in self.toc if level <= max_level
        ]
        if not entries:
            return

        toc = ["## Table of Contents"]
        for title, level, locator in entries:
            toc.append(f"1. [{title}](#{locator})")

        insert_loc = 1 if self.has_title else 0
        self.markdown.insert(insert_loc, "\n".join(toc) + "\n\n")

    def add_table(self, data, use_header: bool = True) -> None:
        """Generate an HTML table from a 2D array-like structure."""
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
        self.markdown.append("\n".join(table) + "\n\n")

    def to_markdown(self) -> str:
        """Return the whole report as a single Markdown string."""
        return "".join(self.markdown)

    def save(self, out_path: Union[str, Path]) -> None:
        """Write the Markdown report to disk."""
        out_path = Path(out_path)
        out_path.write_text(self.to_markdown(), encoding="utf-8")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_image_aspect(img_path: Union[str, Path]) -> Optional[float]:
    """Return width/height aspect ratio for an image, or None on failure."""
    try:
        img_path = Path(img_path)
        with Image.open(img_path) as im:
            w, h = im.size
        if h == 0:
            return None
        return float(w) / float(h)
    except Exception as exc:  # pragma: no cover - best-effort helper
        log.debug(f"Could not get image size for {img_path}: {exc}")
        return None


def width_fraction_from_aspect(aspect: Optional[float], ncols: int = 1) -> float:
    """
    Decide how much of the text width a figure should occupy, based on
    its aspect ratio (width / height) and the number of columns.

    Returns a fraction in (0, 1], e.g. 0.6 = 60% of text width.
    """
    # Multi-column layout: let each column take ~1/ncols of the width.
    if ncols > 1:
        base = 1.0 / float(ncols)
        return min(1.0, base * 0.85)

    if aspect is None:
        return 0.7

    # Very tall (height >> width)
    if aspect < 0.8:
        return 0.6

    # Roughly square to moderately rectangular
    if aspect < 1.4:
        return 0.75

    # Very wide (width >> height)
    return 0.9


def markdown_to_pdf(
    markdown_source: Union[str, Path],
    pdf_path: Union[str, Path],
    base_url: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Render Markdown (string or file) to PDF.

    Args:
        markdown_source: Markdown string or path to a .md file.
        pdf_path: Output PDF path.
        base_url: Base directory for resolving relative image paths, CSS, etc.
    """
    pdf_path = Path(pdf_path)

    if isinstance(markdown_source, Path):
        markdown_text = markdown_source.read_text(encoding="utf-8")
        if base_url is None:
            base_url = markdown_source.parent
    else:
        markdown_text = markdown_source

    if base_url is None:
        base_url = pdf_path.parent
    base_url = Path(base_url)

    # Relative paths to the bundled fonts, normalized to POSIX-style
    font_regular_rel = os.path.relpath(FONT_REGULAR_PATH, base_url).replace(os.sep, "/")
    font_bold_rel = os.path.relpath(FONT_BOLD_PATH, base_url).replace(os.sep, "/")

    body_html = _MD.render(markdown_text)

    full_html = (
        "<!doctype html>\n"
        "<html>\n"
        "<head>\n"
        "  <meta charset='utf-8'>\n"
        "  <style>\n"
        "    @page {\n"
        "      size: A4;\n"
        "        margin: 1.5cm 1.8cm 1.8cm 1.8cm;;\n"
        "    }\n"
        "    @font-face {\n"
        "      font-family: 'FloatSans';\n"
        f"      src: url('{font_regular_rel}') format('truetype');\n"
        "      font-weight: 400;\n"
        "      font-style: normal;\n"
        "    }\n"
        "    @font-face {\n"
        "      font-family: 'FloatSans';\n"
        f"      src: url('{font_bold_rel}') format('truetype');\n"
        "      font-weight: 700;\n"
        "      font-style: normal;\n"
        "    }\n"
        "    body {\n"
        "      font-family: 'FloatSans',\n"
        "        -apple-system, BlinkMacSystemFont,\n"
        "        'Segoe UI', 'Helvetica Neue', Helvetica, Arial,\n"
        "         sans-serif;\n"
        "      font-size: 11pt;\n"
        "      line-height: 1.4;\n"
        "    }\n"
        "    img { max-width: 100%; height: auto; }\n"
        "    img.figure-img { display: block; margin: 0.5em auto; }\n"
        "    h1 { font-size: 18pt; margin: 0 0 0.4em 0; font-weight: 700; }\n"
        "    h2 { font-size: 14pt; margin: 1.0em 0 0.4em 0; font-weight: 700; }\n"
        "    h3 { font-size: 12pt; margin: 0.8em 0 0.3em 0; font-weight: 700; }\n"
        "    table { width: 100%; border-collapse: collapse; }\n"
        "    th, td { padding: 0.25em 0.4em; }\n"
        "  </style>\n"
        "</head>\n"
        "<body>\n"
        f"{body_html}\n"
        "</body>\n"
        "</html>\n"
    )
    HTML(string=full_html, base_url=str(base_url)).write_pdf(str(pdf_path))

    return pdf_path
