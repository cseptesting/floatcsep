.. _experiment_config:

Experiment Configuration
========================

**floatCSEP** provides a standardized workflow for forecasting experiments, the instructions of which can be set in a configuration file. Here, we need to define the Experiments' temporal settings, geographic region, seismic catalogs, forecasting models, evaluation tests and any post-process options.


Configuration Structure
-----------------------
Configuration files are written in ``YAML`` format and are divided into different aspects of the experiment setup:

1. **Metadata**: Experiment's information such as its ``name``, ``authors``, ``doi``, ``URL``, etc.
2. **Temporal Configuration** (``time_config``): Temporal characteristics of the experiment, such as the start and end dates, experiment class (time-independent or time-dependent), the testing intervals, etc.
3. **Spatial and Magnitude Configuration** (``region_config``): Describes the testing region, such as its geographic bounds, magnitude ranges, and depth ranges.
4. **Seismic Catalog** (``catalog``): Defines the seismicity data source to test the models. It can either link to a seismic network API, or an existing file in the system.
5. **Models** (``models``): Configuration of forecasting models. It can direct to an additional configuration file with ``model_config`` for readability. See :ref:`model_config`.
6. **Evaluation Tests** (``tests``): Configuration of the statistical tests to evaluate the models. It can direct to an additional configuration file with ``test_config`` for readability. See :ref:`evaluation_config`.
7. **Postprocessing** (``postprocess``): Instructions on how to process and visualize the experiment's results, such as plotting forecasts or generating reports. See :ref:`postprocess`.

.. note::

    `YAML` (Yet Another Markup Language) is a human-readable format used for configuration files. It uses **key: value** pairs to define settings, and indentation to represent nested structures. Lists are denoted by hyphens (`-`).


**Example Basic Configuration**:

.. code-block:: yaml
   :caption: config.yml

   name: CSEP Experiment
   time_config:
     start_date: 2010-1-1T00:00:00
     end_date: 2020-1-1T00:00:00
   region_config:
     region: region.txt
     mag_min: 4.0
     mag_max: 9.0
     mag_bin: 0.1
     depth_min: 0
     depth_max: 70
   catalog: catalog.csv
   models:
     - Smoothed-Seismicity:
         path: ssm.dat
     - Uniform:
         path: uniform.dat
   tests:
     - Poisson S-test:
         func: poisson_evaluations.spatial_test
         plot_func: plot_poisson_consistency_test
   postprocess:
     plot_forecasts:
       cmap: magma
       catalog: True



Experiment Metadata
-------------------

.. list-table::
   :widths: 20 80

   * - **name**
     - Maximum magnitude to be considered.
   * - **authors**
     - Authors of the experiment
   * - **doi**
     - identifier associated to the experiment
   * - **URL**
     - repository of the experiment (e.g., Github)

Any non-parsed parameter (e.g., not specified in the documentation) will be stored also as metadata.


Temporal Definition
-------------------

Configuring the experiment temporal definition is indicated with the ``time_config`` option, followed by an indented block of admissible parameters. The purpose of this configuration section is to set a testing **time-window**, or a sequence of **time-windows**. Each time-window is defined by two ``datetime`` strings representing its lower and upper edges.

Time-windows can be defined from different combination of the following parameters:

.. list-table::
   :widths: 20 80

   * - **start_date**
     - The start date of the experiment in UTC and ISO8601 format (``%Y-%m-%dT%H:%M:%S``)
   * - **end_date**
     - The end date of the experiment  in UTC and ISO8601 format (``%Y-%m-%dT%H:%M:%S``)
   * - **intervals**
     - An integer amount of testing intervals (time windows). If **horizon** is given, each time-window has a length equal to **horizon**. Else, the range between **start_date** and **end_date** is equally divided into the amount of **intervals**.
   * - **horizon**
     - Indicates the time windows `length`. It is written as a number, followed by a hyphen (`-`) and a time unit (``days``, ``weeks``, ``months``, ``years``). e.g.: ``1-days``, ``2.5-years``.
   * - **growth**
     - How to discretize the time-windows between ``start_date`` and ``end_date``. Options are: **incremental** (The end of a time window matches the beginning of the next) or **cumulative** (All time-windows have a start at the experiment ``start_date``).
   * - **offset**
     - Offset between consecutive time-windows. If none given or ``offset=horizon``, time-windows are non-overlapping. It is written as a number, followed by a hyphen (`-`) and a time unit (``days``, ``weeks``, ``months``, ``years``). e.g.: ``1-days``, ``2.5-years``.
   * - **exp_class**
     - Experiment temporal class. Options are:
       **ti** (default): Time-Independent; **td**: Time-Dependent.

.. note::

    For a Time-Independent (``ti``) experiment class, the following argument combinations are possible:

    - (``start_date``, ``end_date``)
    - (``start_date``, ``end_date``, ``intervals``)
    - (``start_date``, ``end_date``, ``horizon``)
    - (``start_date``, ``intervals``, ``horizon``)

    For a Time-Dependent (``td``) experiment class, the following argument combinations are possible:

    - (``start_date``, ``end_date``, ``intervals``)
    - (``start_date``, ``end_date``, ``horizon``)
    - (``start_date``, ``intervals``, ``horizon``)
    - (``start_date``,  ``end_date``, ``horizon``, ``offset``)
    - (``start_date``,  ``intervals``, ``horizon``, ``offset``)


Some example of parameter combinations:

+------------------------------------------------+----------------------------------------------------------+
| .. code-block:: yaml                           | Two time-windows of equal size between 2010 and 2020     |
|                                                |                                                          |
|    time_config:                                | - ``2010-01-01T00:00:00`` - ``2015-01-01T00:00:00``      |
|        start_date: 2010-01-01T00:00:00         | - ``2015-01-01T00:00:00`` - ``2020-01-01T00:00:00``      |
|        end_date: 2020-01-01T00:00:00           |                                                          |
|        intervals: 2                            |                                                          |
+------------------------------------------------+----------------------------------------------------------+
| .. code-block:: yaml                           | Two cummulative time-windows between 2010 and 2020       |
|                                                |                                                          |
|    time_config:                                | - ``2010-01-01T00:00:00`` - ``2015-01-01T00:00:00``      |
|        start_date: 2010-01-01T00:00:00         | - ``2010-01-01T00:00:00`` - ``2020-01-01T00:00:00``      |
|        end_date: 2020-01-01T00:00:00           |                                                          |
|        intervals: 2                            |                                                          |
|        growth: cumulative                      |                                                          |
+------------------------------------------------+----------------------------------------------------------+
| .. code-block:: yaml                           | Time-Dependent experiment with three ``1-day`` windows   |
|                                                |                                                          |
|    time_config:                                |                                                          |
|        start_date: 2010-01-01T00:00:00         | - ``2010-01-01T00:00:00`` - ``2010-01-02T00:00:00``      |
|        intervals: 3                            | - ``2010-01-02T00:00:00`` - ``2010-01-03T00:00:00``      |
|        horizon: 1-days                         | - ``2010-01-03T00:00:00`` - ``2010-01-04T00:00:00``      |
|        exp_class: td                           |                                                          |
+------------------------------------------------+----------------------------------------------------------+
| .. code-block:: yaml                           | Two overlapping ``7-days`` time-windows                  |
|                                                |                                                          |
|    time_config:                                | - ``2010-01-01T00:00:00`` - ``2010-01-08T00:00:00``      |
|        start_date: 2010-01-01T00:00:00         | - ``2010-01-02T00:00:00`` - ``2020-01-09T00:00:00``      |
|        intervals: 2                            |                                                          |
|        horizon: 7-days                         |                                                          |
|        offset: 1-day                           |                                                          |
|        exp_class: td                           |                                                          |
+------------------------------------------------+----------------------------------------------------------+

Alternatively, time windows can be defined explicitly as a **list** of time-windows (each of which is a **list** of ``datetimes``):

.. code-block:: yaml

    time_config:
      timewindows:
        - - 2010-01-01T00:00:00
          - 2011-01-01T00:00:00
        - - 2011-01-01T00:00:00
          - 2012-01-01T00:00:00

Spatial and Magnitude Definition
--------------------------------

Configuring the spatial and magnitude definitions is done through the ``region_config`` option, followed by an indented block of admissible parameters. Here, we need to define the spatial region (check the `Region <https://docs.cseptesting.org/concepts/regions.html>`_ documentation from **pyCSEP**), the magnitude `bins` (i.e., discretization) and the `depth` extent.

.. list-table::
   :widths: 20 80

   * - **region**
     - The spatial domain where forecasts will be tested. Either a file or a **CSEP** region.
   * - **mag_min**
     - The minimum magnitude of the experiment.
   * - **mag_max**
     - The maximum magnitude of the experiment.
   * - **mag_bin**
     - The size of the magnitude bin.
   * - **depth_min**
     - The minimum depth (in `km`) of the experiment.
   * - **depth_max**
     - The maximum depth (in `km`) of the experiment.


1. The ``region`` parameter can be defined from:


   * A **CSEP** region: They correspond to pre-established testing regions for seismic areas. This parameter is linked to **pyCSEP** functions, and can be one of the following values:

      * ``italy_csep_region``
      * ``nz_csep_region``
      * ``california_relm_region``
      * ``global_region``.

   * A text file with the spatial cells collection. Each cell is defined by its origin (e.g., the x (lon) and y (lat) of the lower-left corner). For example, for a region consisting of three cells, their origins can be written as:

      .. code-block::

          10.0 40.0
          10.0 40.1
          10.1 40.0

   See the **pyCSEP** `Region documentation <https://docs.cseptesting.org/concepts/regions.html#cartesian-grid>`_, the class :class:`~csep.core.regions.CartesianGrid2D` and its method :meth:`~csep.core.regions.CartesianGrid2D.from_origins` for more info.
2. Magnitude definition: We need to define a magnitude discretization or `bins`. The parameters **mag_min**, **mag_max**, **mag_bin** allows to create an uniformly distributed set of bins. For example, the command:

   .. code-block:: yaml

        mag_min: 4.0
        mag_max: 5.0
        mag_bin: 0.5

   would result in two magnitude bins with ranges ``[4.0, 4.5)`` and ``[4.5, 5.0)``. Alternatively, magnitudes can be written explicitly by their bin `left` edge. For example:

   .. code-block:: yaml

      magnitudes:
          - 4.0
          - 4.1
          - 4.2

   resulting in the ``[4.0, 4.1)``, ``[4.1, 4.2)`` and ``[4.2, 4.3)``.


3. Depths: The minimum and maximum depths are just required to filter out seismicity outside those ranges.


Some example of region configurations would be:

+------------------------------------------------+---------------------------------------------------------------------+
| .. code-block:: yaml                           |  - Uses the **CSEP** Italy region, as defined by the function       |
|                                                |    :func:`~csep.core.regions.italy_csep_region`.                    |
|    region_config:                              |  - Discretizes the magnitude range into 40 bins between 4.0 and 9.0 |
|        region: italy_csep_region               |  - Test the models against `crustal` seismicity above 30 km.        |
|        mag_min: 5.0                            |    The -2 is meant in case of shallow events above sea level        |
|        mag_max: 9.0                            |                                                                     |
|        mag_bin: 0.1                            |                                                                     |
|        depth_min: -2                           |                                                                     |
|        depth_max: 30                           |                                                                     |
+------------------------------------------------+---------------------------------------------------------------------+
| .. code-block:: yaml                           | - Loads a file ``region_file.txt`` which contains the cells'        |
|                                                |   originsof the region.                                             |
|    region_config:                              | - Contains two magnitude bins: ``[6.0, 7.0)``, ``[7.0, 8.0)`` and   |
|        region: region_file.txt                 |                                                                     |
|        depth_min: 70                           |                                                                     |
|        depth_max: 150                          |                                                                     |
|        magnitudes:                             |                                                                     |
|            - 6.0                               |                                                                     |
|            - 7.0                               |                                                                     |
|            - 8.0                               |                                                                     |
+------------------------------------------------+---------------------------------------------------------------------+


Seismicity Catalog
------------------

The seismicity catalog can be defined with the ``catalog`` parameter. It represents the **main catalog** of the experiment, and will be used to test the forecasts against, or if required, as input catalog for time-dependent models. It can be obtained from:

* **Authorative data source**

  **floatCSEP** can retrieve the catalog from a seismic network API. The possible options are:

  - ``query_gcmt``: Global Centroid Moment Tensor Catalog (https://www.globalcmt.org/), obtained via ISC (https://www.isc.ac.uk/)
  - ``query_comcat``: ANSS ComCat (https://earthquake.usgs.gov/data/comcat/)
  - ``query_bsi``: Bollettino Sismico Italiano (https://bsi.ingv.it/)
  - ``query_gns``: GNS GeoNet New Zealand Catalog (https://www.geonet.org.nz/)

* **Catalog file in pyCSEP format**

  A file can be used as **main catalog**. It must be in a **pyCSEP** format, namely in the :meth:`~pycsep.utils.readers.csep_ascii` style (see :doc:`pycsep:concepts/catalogs`) or ``.json`` format. The latter is the default catalog used by **floatCSEP**, as it allows the storage of metadata.

  .. note::
      A catalog can be stored as ``.json`` with :meth:`CSEPCatalog.write_json() <csep.core.catalogs.CSEPCatalog.write_json>` using **pyCSEP**.

.. important::
  The main catalog will be stored, and consecutively filtered to the extent of each testing time-window, as well as to the experiment's spatial domain, and magnitude- and depth- ranges.
