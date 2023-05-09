Experiment A - Simple Forecast and Catalog
==========================================

.. currentmodule:: fecsep

.. contents::
    :local:

.. admonition:: **TL; DR**

    In a terminal, navigate to ``fecsep/examples/case_a`` and type:

    .. code-block:: shell

        $ fecsep run config.yml

    After the calculation is complete, the results will be summarized in ``results/report.md``.


Artifacts
---------

The following example shows the definition of a testing experiment of a simple
forecast against a simple catalog. The input structure of the experiment is:

::

    case_a
        ├── catalog.csep
        ├── config.yml
        ├── best_model.dat
        └── region.txt


The testing region consists of a grid with two 1ºx1º bins, whose bottom-left nodes are defined in the file `region.txt`. The grid spacing is obtained automatically. The nodes are:

.. literalinclude:: ../../examples/case_a/region.txt

The testing catalog contains only one event and is formatted in the :meth:`~pycsep.utils.readers.csep_ascii` style (see :doc:`pycsep:concepts/catalogs`). Catalog formats are detected automatically

.. literalinclude:: ../../examples/case_a/catalog.csep


The forecast to be evaluated is written in the ``.dat`` format (:doc:`pycsep:concepts/forecasts`). Forecast formats are detected automatically (see :class:`fecsep.readers`)

.. literalinclude:: ../../examples/case_a/best_model.dat

Configuration
-------------

The experiment is defined by a time-, region-, model- and evaluation-configurations. In this example, they are written together in the ``config.yml`` file.


Time
~~~~

    The time configuration is manifested in the ``time-config`` inset. The simplest definition is to set only the start and end dates of the experiment. These are always UTC time in isoformat (``%Y-%m-%dT%H:%M:%S.%f`` - ISO861):

    .. literalinclude:: ../../examples/case_a/config.yml
       :language: yaml
       :lines: 3-5

    .. note::

        In case the time window are bounded by their midnights, the ``start_date`` and ``end_date`` can be in the format ``%Y-%m-%d``.

    The results of the experiment run will be associated with this time window, whose identifier will be its bounds: ``2020-01-01_2021-01-01``

Region
~~~~~~

    The region - a file path or :class:`csep` function (e.g. :obj:`csep.core.regions.italy_csep_region`) -, the depth limits and magnitude  discretization are defined in the ``region_config`` inset.

    .. literalinclude:: ../../examples/case_a/config.yml
       :language: yaml
       :lines: 7-13


Catalog
~~~~~~~

    It is defined in the ``catalog`` inset. This should only make reference to a catalog file or a catalog query function (e.g. ``query_comcat``). ``fecsep`` will automatically filter the catalog to the experiment time, spatial and magnitude frames:

    .. literalinclude:: ../../examples/case_a/config.yml
       :language: yaml
       :lines: 15-15

Models
~~~~~~
    The model configuration is set in the ``models`` inset with a list of model names, which specify their file paths (and other attributes). Here, we just set the path as ``best_model.dat``, whose format is automatically detected.

    .. literalinclude:: ../../examples/case_a/config.yml
       :language: yaml
       :lines: 17-19

Evaluations
~~~~~~~~~~~
    The experiment's evaluations are defined in the ``tests`` inset. It should be a list of test names, making reference to their function and plotting function. These can be either defined in ``pycsep`` (see :doc:`pycsep:concepts/evaluations`) or manually. In this example, we employ the consistency N-test: its function is :func:`csep.core.poisson_evaluations.number_test`, whereas its plotting function correspond to :func:`csep.utils.plots.plot_poisson_consistency_test`

.. literalinclude:: ../../examples/case_a/config.yml
   :language: yaml
   :lines: 21-24


Running the experiment
----------------------

Run command
~~~~~~~~~~~

    The experiment can be run by simply navigating to the ``examples/case_a`` folder in the terminal and typing.

    .. code-block:: shell

        fecsep run config.yml

    This will automatically set all the calculation paths (testing catalogs, evaluation results, figures) and will create a summarized report in ``results/report.md``.

Plot command
~~~~~~~~~~~~

    If only the result plots are desired, when the calculation was already completed, you can type:

    .. code-block:: shell

        fecsep plot config.yml

    This can be used, for example, when an additional plot is desired. Try adding to ``config.yml`` the following lines

    .. code-block:: yaml

        postproc_config:
          plot_forecasts: True

    and re-run with the ``plot`` command. A forecast figure will appear in ``results/{window}/forecasts``

Results
~~~~~~~

    The :obj:`~fecsep.cmd.main.run` command creates the result path tree for each time window analyzed.

    *  The testing catalog of the window is stored in ``results/{window}/catalog``  in ``json`` format. This is a subset of the global testing catalog.
    *  Human-readable results are found in ``results/{window}/evaluations``
    *  Catalog and evaluation results figures in ``results/{window}/figures``.
    *  The complete results are summarized in ``results/report.md``


Advanced
--------

The experiment run logic can be seen in the file ``case_a.py``, which executes the same example but in python source code. The run logic of the terminal commands ``run``, ``plot`` and ``reproduce`` can be found in :class:`fecsep.cmd.main`


