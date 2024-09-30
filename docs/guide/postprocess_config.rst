.. _postprocess:

Post-Process Options
====================

The ``postprocess`` inset can be used within an experiment configuration file to configure the **plotting** and **reporting** functions to be performed after the experiment calculations have been completed. The plotting functions provide a graphic representation of the catalogs, forecasts and evaluation results, whereas the reporting functions assemble these into a human-readable report.

**Example postprocess configuration**:

.. code-block:: yaml
   :caption: config.yml
   :emphasize-lines: 8-

    name: experiment
    time_config: ...
    region_config: ...
    catalog: ...
    model_config: ...
    test_config: ...

    postprocess:
      plot_forecasts:
        colormap: magma
        basemap: ESRI_terrain
        catalog: True

      plot_catalog:
        basemap: google-satellite
        mag_ticks: [5, 6, 7, 8]
        markersize: 7


.. important::

    By default, **floatCSEP** plots the testing catalogs, forecasts and results, summarizing them into a **Markdown** report. The postprocess configuration aids to customize these options, or to extend them by using custom python scripts.

Plot Forecasts
--------------

**floatCSEP** can quickly plot the spatial rates of used and/or created forecasts. The ``plot_forecast`` command wraps the functionality of the **pyCSEP** function :func:`~csep.utils.plots.plot_spatial_dataset`, used to plot mean rates (in ``log10``) of both :class:`~csep.core.forecasts.GriddedForecast` and :class:`~csep.core.forecasts.CatalogForecast`.
Most arguments of ``plot_forecast`` mimics those of :func:`~csep.utils.plots.plot_spatial_dataset` with some extra additions. These are summarized here:

.. dropdown::  Forecast plotting arguments
   :animate: fade-in-slide-down
   :icon: list-unordered

   .. list-table::
      :widths: 20 80
      :header-rows: 1

      * - **Arguments**
        - **Description**
      * - ``all_time_windows``
        - Whether all testing time windows are plotted (true or false). By default, only the last time window is plotted.
      * - ``figsize``
        - List with the figure proportions. Default is `[6.4, 4.8]`
      * - ``title``
        - Title for the plot. Default is None
      * - ``title_size``
        - Size of the title text. Default is 10
      * - ``projection``
        - Projection for the map. Default ``cartopy.crs.PlateCarree`` Example:

          .. code-block:: yaml

            plot_forecasts:
                projection: Mercator

          or if the projection contains keyword arguments:

          .. code-block:: yaml

            plot_forecasts:
                projection:
                    Mercator:
                        central_longitude: 50

      * - ``grid``
        - Whether to show grid lines. Default is True
      * - ``grid_labels``
        - Whether to show grid labels. Default is True
      * - ``grid_fontsize``
        - Font size for grid labels. Default is 10.0
      * - ``basemap``
        - Basemap option. Possible values are: ``stock_img``, ``google-satellite``,  ``ESRI_terrain``, ``ESRI_imagery``, ``ESRI_relief``, ``ESRI_topo``, ``ESRI_terrain``, or a webservice URL. Default is None
      * - ``coastline``
        - Flag to plot coastline. Default is True
      * - ``borders``
        - Flag to plot country borders. Default is False
      * - ``region_border``
        - Flag to plot the forecast region border. Default is True
      * - ``tile_scaling``
        - Zoom level (1-12) for basemap tiles or ``auto`` for automatic scaling
      * - ``linewidth``
        - Line width of borders and coastlines. Default is 1.5
      * - ``linecolor``
        - Color of borders and coastlines. Default is ``black``
      * - ``cmap``
        - Color map for the plot. Default is ``viridis``
      * - ``clim``
        - Range of the colorbar, in ``log10`` values. Example: ``[-5, 0]``
      * - ``clabel``
        - Label for the colorbar. Default is None
      * - ``clabel_fontsize``
        - Font size for the colorbar label. Default is None
      * - ``cticks_fontsize``
        - Font size for the colorbar ticks. Default is None
      * - ``alpha``
        - Transparency level. Default is 1
      * - ``alpha_exp``
        - Exponent for the alpha function, recommended between 0.4 and 1. Default is 0
      * - ``catalog``
        - Plots the testing catalog on top of the forecast, corresponding to the forecast time window.

          .. code-block:: yaml

            plot_forecasts:
                catalog: True

          and if the catalog needs to be customized:

          .. code-block:: yaml

            plot_forecasts:
                catalog:
                  legend_loc: 1
                  legend_fontsize: 14
                  markercolor: blue

          See :ref:`plot_catalogs` for customization options.


.. important::

    By default, only the forecast corresponding to the last time window of a model is plotted. To plot all time windows, use ``all_time_windows: True``


.. _plot_catalogs:

Plot Catalogs
-------------

Test catalogs are automatically plotted when **floatCSEP** calculations are finished. Similar to plotting the forecasts, the ``plot_catalog`` command wraps the functionality of the **pyCSEP** function :func:`~csep.utils.plots.plot_catalog`.



.. dropdown::  Catalog Plotting Arguments
   :animate: fade-in-slide-down
   :icon: list-unordered

   .. list-table::
      :widths: 20 80
      :header-rows: 1

      * - **Arguments**
        - **Description**
      * - ``all_time_windows``
        - If along the main testing catalogs, all sub-testing catalogs from all the time windows are plotted (true or false). Default is False.
      * - ``figsize``
        - List or tuple with the figure proportions. Default is [6.4, 4.8].
      * - ``title``
        - Title for the plot. Default is the catalog name.
      * - ``title_size``
        - Size of the title text. Default is 10.
      * - ``filename``
        - File name to save the figure. Default is None.
      * - ``projection``
        - Projection for the map. Default ``cartopy.crs.PlateCarree`` Example:

          .. code-block:: yaml

            plot_forecasts:
                projection: Mercator

          or if the projection contains keyword arguments:

          .. code-block:: yaml

            plot_forecasts:
                projection:
                    Mercator:
                        central_longitude: 50

      * - ``basemap``
        - Basemap option. Possible values are: ``stock_img``, ``google-satellite``,  ``ESRI_terrain``, ``ESRI_imagery``, ``ESRI_relief``, ``ESRI_topo``, ``ESRI_terrain``, or a webservice URL. Default is None
      * - ``coastline``
        - Flag to plot coastline. Default is True.
      * - ``grid``
        - Whether to display grid lines. Default is True.
      * - ``grid_labels``
        - Whether to display grid labels. Default is True.
      * - ``grid_fontsize``
        - Font size for grid labels. Default is 10.0.
      * - ``marker``
        - Marker type for plotting earthquakes.
      * - ``markersize``
        - Constant size for all earthquakes.
      * - ``markercolor``
        - Color for all earthquakes. Default is ``blue``.
      * - ``borders``
        - Flag to plot country borders. Default is False.
      * - ``region_border``
        - Flag to plot the catalog region border. Default is True.
      * - ``alpha``
        - Transparency level for the earthquake scatter. Default is 1.
      * - ``mag_scale``
        - Scaling factor for the scatter plot based on earthquake magnitudes.
      * - ``legend``
        - Flag to display the legend box. Default is True.
      * - ``legend_loc``
        - Position of the legend (integer or string). Default is 'best'.
      * - ``mag_ticks``
        - List of magnitude ticks to display in the legend.
      * - ``labelspacing``
        - Separation between legend ticks. Default is 0.5.
      * - ``tile_scaling``
        - Zoom level (1-12) for basemap tiles, or ``auto`` for automatic scaling based on extent.
      * - ``linewidth``
        - Line width of borders and coastlines. Default is 1.5.
      * - ``linecolor``
        - Color of borders and coastlines. Default is ``black``.


.. important::

    By default, only the main test catalog (containing all events within the experiment frame) is plotted. To also plot the test catalogs from each time window separately, use ``all_time_windows: True``


Custom Plotting
---------------

Additional plotting functionality can be injected to an experiment by using a custom **python** script, which is specified within the ``postprocess`` configuration:

**Example:**

.. code-block:: yaml

    postprocess:
      plot_custom: plot_script.py:main

where the script path and a function within should be written as:

.. code-block:: yaml

    plot_custom: {python_script_path}:{function_name}

This option provides a `hook` for python code to be run after the experiment calculation, giving it read access to attributes from the :class:`floatcsep.experiment.Experiment` class. The `hook` requirements are that the script to be located within the same directory as the configuration file, whereas the function must receive a :class:`floatcsep.experiment.Experiment` as unique argument:


**Example custom plot script**:

.. code-block:: python

    from floatcsep import Experiment

    def main_function(experiment: Experiment):

        timewindows = experiment.timewindows
        model = experiment.get_model("pymock")

        rates = []
        start_times = []

        for timewindow in timewindows:
            forecast = model.get_forecast(timewindow)
            rates.append(forecast.event_counts)
            start_times = timewindow[0]

        fig, ax = plt.subplots(1, 1)
        ax.plot(start_times, rates)
        pyplot.savefig("results/pymock_rates.png")


In this way, the plot function can use all the :class:`~floatcsep.experiment.Experiment` attributes/methods to access catalogs, forecasts and test results. Please check the :ref:`postprocess_api` and the Tutorial :ref:`case_g` for an advanced use.



.. _custom_reporting:

Custom Reporting
----------------

In addition to plotting, **floatCSEP** allows users to generate custom reports in **Markdown** format. The **MarkdownReport** class is designed to support the automatic creation of these reports, allowing users to assemble figures, text, and other results in a well-structured manner.

The custom report functionality can be invoked by specifying the following in the ``postprocess`` configuration:

**Example**:

.. code-block:: yaml
   :caption: config.yml

    postprocess:
      report: report_script.py:generate_report

This configuration specifies a custom **python** script with the following format:

.. code-block:: yaml

    report: {python_script_path}:{function_name}

The script must be located within the same directory as the configuration file and the function must receive an instance of :class:`floatcsep.experiment.Experiment` instance as its only argument.

**Example Custom Report Script:**:

.. code-block:: python

    from floatcsep.utils.reporting import MarkdownReport

    def generate_report(experiment):
        # Create a MarkdownReport object
        report = MarkdownReport(out_name="custom_report.md")

        # Add an introduction based on the experiment details
        intro = {
            'simulation_name': experiment.name,
            'forecast_name': 'ETAS',
            'origin_time': experiment.start_date,
            'evaluation_time': experiment.end_date,
            'catalog_source': 'Observed Catalog',
            'num_simulations': 10000
        }
        report.add_introduction(intro)

        # Add some text
        report.add_text(['This report contains results from the ETAS model experiment.', 'Additional details below.'])


        # Add a figure (for example, forecast rates over time)
        report.add_figure(
            title="Forecast Rates",
            relative_filepaths=["results/2020-01-01_2020_01_02/forecasts/etas/forecast_rates.png"],
            ncols=1,
            caption="Forecasted seismicity rates over time."
        )

        # Save the report
        report.save(save_dir="results")


The **MarkdownReport** class provides various methods for assembling a report, allowing the user to format the content, insert figures, add tables, and generate text dynamically based on the results of an experiment.

For more advanced usage of report generation, please review the `default` **floatCSEP** report in the module :mod:`floatcsep.postprocess.reporting.generate_report`, an implementation example in the tutorial :ref:`case_h` and the :ref:`postprocess_api` for an advance use.


.. _postprocess_api:

Postprocess API
---------------

Here are some basic functionalities from **floatCSEP** to access catalogs, forecasts and results using **python** code:

.. dropdown:: Experiment and Catalogs
   :animate: fade-in-slide-down
   :icon: list-unordered

   .. list-table::
      :widths: 20 80
      :header-rows: 1

      * - **Method/Attribute**
        - **Description**
      * - :attr:`Experiment.timewindows <floatcsep.experiment.Experiment>`
        - A list of timewindows, where each is a pair of :class:`datetime.datetime` objects representing the window boundaries.
      * - :attr:`Experiment.start_date <floatcsep.experiment.Experiment>`
        - The starting :class:`datetime.datetime` of the experiment.
      * - :attr:`Experiment.end_date <floatcsep.experiment.Experiment>`
        - The end :class:`datetime.datetime` of the experiment.
      * - :attr:`Experiment.region <floatcsep.experiment.Experiment>`
        - A :class:`csep.core.regions.CartesianGrid2D` object representing the spatial extent of the experiment.
      * - :attr:`Experiment.mag_min <floatcsep.experiment.Experiment>`
        - The minimum magnitude of the experiment.
      * - :attr:`Experiment.mag_max <floatcsep.experiment.Experiment>`
        - The maximum magnitude of the experiment.
      * - :attr:`Experiment.mag_bin <floatcsep.experiment.Experiment>`
        - The magnitude bin size.
      * - :attr:`Experiment.magnitudes <floatcsep.experiment.Experiment>`
        - A list of the magnitude bins of the experiment.
      * - :attr:`Experiment.depth_min <floatcsep.experiment.Experiment>`
        - The minimum depth of the experiment.
      * - :attr:`Experiment.depth_max <floatcsep.experiment.Experiment>`
        - The maximum depth of the experiment.
      * - :attr:`Experiment.run_dir <floatcsep.experiment.Experiment>`
        - Returns the running directory of the experiment, where all evaluation results and figures are stored. Default is ``results/`` unless specified different in the ``config.yml``.
      * - :attr:`Experiment.models <floatcsep.experiment.Experiment>`
        - Returns a list containing all the experiment's :class:`~floatcsep.model.Model` objects.
      * - :meth:`Experiment.get_model(str) <floatcsep.experiment.Experiment.get_model>`
        - Returns a :class:`~floatcsep.model.Model` from its given name.
      * - :attr:`Experiment.tests <floatcsep.experiment.Experiment>`
        - Returns a list containing all the experiment's :class:`~floatcsep.evaluation.Evaluation` objects.
      * - :meth:`Experiment.get_test(str) <floatcsep.experiment.Experiment.get_test>`
        - Returns a :class:`~floatcsep.evaluation.Evaluation` from its given name
      * - :attr:`Experiment.catalog_repo <floatcsep.infrastructure.repositories.CatalogRepository>`
        - A :class:`~floatcsep.infrastructure.repositories.CatalogRepository` which can access the experiments catalogs.
      * - :attr:`Experiment.catalog_repo.catalog <floatcsep.experiment.Experiment>`
        - The main catalog of the experiment, of :class:`csep.core.catalogs.CSEPCatalog` class.
      * - :meth:`Experiment.catalog_repo.get_test_cat(timewindow) <floatcsep.infrastructure.repositories.CatalogRepository.get_test_cat>`
        - Returns the testing catalog for a given ``timewindow`` formatted as string. Use :func:`floatcsep.utils.helpers.timewindow2str` in case the window is a list of two  :class:`datetime.datetime` objects.



.. dropdown::  Models and forecasts
   :animate: fade-in-slide-down
   :icon: list-unordered

   The experiment models can be accessed by using :attr:`Experiment.models <floatcsep.experiment.Experiment>` or :meth:`Experiment.get_model(str) <floatcsep.experiment.Experiment.get_model>`.

   .. list-table::
      :widths: 60 40
      :header-rows: 1

      * - **Method/Attribute**
        - **Description**
      * - :attr:`Model.name <floatcsep.model.Model>`
        - Name of the model
      * - :meth:`Model.get_forecast(timewindow) <floatcsep.model.Model.get_forecast>`
        - Returns the forecast for a given ``timewindow`` (formatted as string. Use :func:`floatcsep.utils.helpers.timewindow2str` in case the window is a list of two :class:`datetime.datetime` objects). Example:

          .. code-block:: python

              model = experiment.get_model('etas')
              timewindow = experiment.timewindows[0]
              timewindow_str = timewindow2str(timewindow)
              model.get_forecast(timewindow_str)

      * - :attr:`Model.registry.path <floatcsep.infrastructure.registries.ForecastRegistry>`
        - Directory of the model file or source code.
      * - :attr:`Model.registry.database <floatcsep.infrastructure.registries.ForecastRegistry>`
        - Database path where forecasts are stored.
      * - :attr:`TimeIndependentModel.forecast_unit <floatcsep.model.TimeIndependentModel>`
        - The forecast unit for a time independent model.
      * - :meth:`TimeDependentModel.func <floatcsep.model.TimeIndependentModel>`
        - The function command to execute a time dependent source code.
      * - :meth:`TimeDependentModel.func_kwargs`
        - The keyword arguments of the model, passed to the arguments file.
      * - :meth:`TimeDependentModel.registry.args_file <floatcsep.infrastructure.registries.ForecastRegistry>`
        - The path of the arguments file. Default is ``args.txt``.
      * - :meth:`TimeDependentModel.registry.input_cat <floatcsep.infrastructure.registries.ForecastRegistry>`
        - The path of the input catalog for the model execution.


.. dropdown::  Results
   :animate: fade-in-slide-down
   :icon: list-unordered

   The experiment evaluations can be accessed by using :attr:`Experiment.tests <floatcsep.experiment.Experiment>` or :meth:`Experiment.get_test(str) <floatcsep.experiment.Experiment.get_test>`.

   .. list-table::
      :widths: 50 50
      :header-rows: 1

      * - **Method/Attribute**
        - **Description**
      * - :meth:`Evaluation.read_results(timewindow, models) <floatcsep.evaluation.Evaluation.read_results>`
        - Returns the evaluation results for a given time window and models. Example usage:

          .. code-block:: python

            test = experiment.get_test('n_test')  # get a test by its name
            model = experiment.get_model('etas')  # get a model by its name
            timewindow = experiment.timewindows[0]  # first time window
            result = test.read_results(timewindow, model)

          or from all models:

          .. code-block:: python

            test = experiment.get_test('s_test')  # get a test by its name
            timewindow = experiment.timewindows[-1]  # last time window
            result = test.read_results(timewindow, experiment.models)



.. dropdown:: MarkdownReport Methods
   :animate: fade-in-slide-down
   :icon: list-unordered

   .. list-table::
      :widths: 20 80
      :header-rows: 1

      * - **Method**
        - **Description**
      * - :meth:`MarkdownReport.add_introduction`
        - Adds an introductory section to the report. This typically contains metadata such as the simulation name, forecast model, evaluation time, and other summary information.
      * - :meth:`MarkdownReport.add_text`
        - Adds text to the report. Each entry corresponds to a paragraph, and the text argument should be provided as a list of strings.
      * - :meth:`MarkdownReport.add_figure`
        - Inserts one or more figures into the report. You can specify the title, filepaths to the figures, and an optional caption. Figures are arranged in rows and columns as specified by the ``ncols`` argument.
      * - :meth:`MarkdownReport.add_table`
        - Creates a table in the report. The table data should be provided as a list of rows, where each row is a list of cell contents.
      * - :meth:`MarkdownReport.add_list`
        - Adds a bulleted list of items to the report.
      * - :meth:`MarkdownReport.add_heading`
        - Inserts a heading into the report. The ``level`` argument controls the heading level (1 for top-level, 2 for subheading, etc.).
      * - :meth:`MarkdownReport.table_of_contents`
        - Generates a table of contents based on the headings and sections included in the report so far. It will be automatically placed at the beginning of the report if an introduction is included.
      * - :meth:`MarkdownReport.save`
        - Saves the Markdown report to a specified directory.

