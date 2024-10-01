.. _case_a:

A - Testing a Simple Model
==========================

The following example shows the definition of a testing experiment of a single **time-independent** forecast against a catalog.

.. currentmodule:: floatcsep

.. admonition:: **TL; DR**

    In a terminal, navigate to ``floatcsep/tutorials/case_a`` and type:

    .. code-block:: console

        $ floatcsep run config.yml

    After the calculation is complete, the results will be summarized in ``results/report.md``.

.. contents:: Contents
    :local:
    :depth: 2


Experiment Components
---------------------

The source code can be found in the ``tutorials/case_a`` folder or in  `GitHub <https://github.com/cseptesting/floatcsep/blob/main/tutorials/case_a>`_. The directory structure of the experiment is:

::

    case_a
        ├── region.txt
        ├── catalog.csep
        ├── best_model.dat
        └── config.yml


* The testing region ``region.txt`` consists of a grid with two 1ºx1º bins, defined by its bottom-left nodes. The grid spacing is obtained automatically. The nodes are:

    .. literalinclude:: ../../tutorials/case_a/region.txt
       :caption: tutorials/case_a/region.txt

* The testing catalog ``catalog.csep`` contains only one event and is formatted in the :meth:`~pycsep.utils.readers.csep_ascii` style (see :doc:`pycsep:concepts/catalogs`). Catalog formats are detected automatically

    .. literalinclude:: ../../tutorials/case_a/catalog.csep
       :caption: tutorials/case_a/catalog.csep

* The forecast ``best_model.dat`` to be evaluated is written in the ``.dat`` format (see :doc:`pycsep:concepts/forecasts`). Forecast formats are detected automatically (see :mod:`floatcsep.utils.readers.ForecastParsers`)

    .. literalinclude:: ../../tutorials/case_a/best_model.dat
        :caption: tutorials/case_a/best_model.dat


Configuration
-------------

The experiment is defined by a time-, region-, model- and test-configurations, as well as a catalog and a region. In this example, they are written together in the ``config.yml`` file.


.. important::

    Every file path (e.g., of a catalog) specified in the ``config.yml`` file should be relative to the directory containing the configuration file.



Time
~~~~

    The time configuration is manifested in the ``time_config`` inset. The simplest definition is to set only the start and end dates of the experiment. These are always UTC date-times in isoformat (``%Y-%m-%dT%H:%M:%S.%f`` - ISO861):

    .. literalinclude:: ../../tutorials/case_a/config.yml
       :caption: tutorials/case_a/config.yml
       :language: yaml
       :lines: 3-5

    .. note::

        In case the time window are bounded by their midnights, the ``start_date`` and ``end_date`` can be in the format ``%Y-%m-%d``.

    The results of the experiment run will be associated with this time window, whose identifier will be its bounds: ``2020-01-01_2021-01-01``

Region
~~~~~~

    The region - a file path or a :mod:`pycsep` function, such as :obj:`~csep.core.regions.italy_csep_region` (check the available regions in :mod:`csep.core.regions`) -, the depth limits and magnitude discretization are defined in the ``region_config`` inset.

    .. literalinclude:: ../../tutorials/case_a/config.yml
       :caption: tutorials/case_a/config.yml
       :language: yaml
       :lines: 7-13


Catalog
~~~~~~~

    It is defined in the ``catalog`` inset. This should only make reference to a catalog **file** or a catalog **query function** (e.g. :func:`~csep.query_comcat`). **floatCSEP** will automatically filter the catalog to the experiment time, spatial and magnitude frames:

    .. literalinclude:: ../../tutorials/case_a/config.yml
       :caption: tutorials/case_a/config.yml
       :language: yaml
       :lines: 15-15

Models
~~~~~~
    The model configuration is set in the ``models`` inset with a list of model names, which specify their file paths (and other attributes). Here, we just set the path as ``best_model.dat``, whose format is automatically detected.

    .. literalinclude:: ../../tutorials/case_a/config.yml
       :caption: tutorials/case_a/config.yml
       :language: yaml
       :lines: 17-19

    .. note::

        A time-independent forecast model has default units of ``[eq/year]`` per cell. A forecast defined for a different number of years can be specified with the ``forecast_unit: {years}`` attribute.

Evaluations
~~~~~~~~~~~
    The experiment's evaluations are defined in the ``tests`` inset. It should be a list of test names making reference to their function and plotting function. These can be either from **pyCSEP** (see :doc:`pycsep:concepts/evaluations`) or defined manually. Here, we use the Poisson consistency N-test: its function is :func:`poisson_evaluations.number_test <csep.core.poisson_evaluations.number_test>` with a plotting function :func:`plot_poisson_consistency_test <csep.utils.plots.plot_poisson_consistency_test>`

.. literalinclude:: ../../tutorials/case_a/config.yml
   :caption: tutorials/case_a/config.yml
   :language: yaml
   :lines: 21-24


Running the experiment
----------------------

Run command
~~~~~~~~~~~

    The experiment can be run by simply navigating to the ``tutorials/case_a`` folder in the terminal and typing.

    .. code-block:: console

        $ floatcsep run config.yml

    This will automatically set all the calculation paths (testing catalogs, evaluation results, figures) and will create a summarized report in ``results/report.md``.

    .. note::

        The command ``floatcsep run {config_file}`` can be called from any working directory, as long as the specified file paths (e.g. region, models) are relative to the ``config.yml`` file.


Results
~~~~~~~

    The :obj:`~floatcsep.cmd.main.run` command creates the result path tree for each time window analyzed.

    *  The testing catalog of the window is stored in ``results/{window}/catalog``  in ``json`` format. This is a subset of the global testing catalog.
    *  Human-readable results are found in ``results/{window}/evaluations``
    *  Catalog, forecasts and evaluation results figures in ``results/{window}/figures``.
    *  The complete results are summarized in ``results/report.md``


Advanced
~~~~~~~~

The experiment run logic can be seen in the file ``case_a.py``, which executes the same example but in python source code. The run logic of the terminal commands ``run``, ``plot`` and ``reproduce`` can be found in :mod:`floatcsep.commands.main`, and can be customized by creating a script similar to ``case_a.py``.


