Experiment A - Simple Forecast and Catalog
==========================================

.. currentmodule:: fecsep

.. contents::


.. admonition:: **TL; DR**

    In a terminal, navigate to ``fecsep/examples/case_a`` and type:

    .. code-block:: shell

        $ fecsep run config.yml

    After the calculation is complete, the results will be summarized in ``results/report.md``.


Experiment's artifacts
----------------------

The following example shows the definition of a testing experiment of a simple
forecast against a simple catalog. The input structure of the experiment is:

::

    case_a
        ├── catalog.csep
        ├── config.yml
        ├── best_model.dat
        └── region.txt


The testing region consists of a grid with two 1ºx1º bins, whose bottom-left nodes are defined in the file `region.txt`. The grid spacing is obtained automatically:

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

    In case the time window are bounded by their midnights, the ``start_date`` and ``end_date`` can be in the format ``%Y-%m-%d``.

    The results of the experiment will be associated with this time window, whose identifier will be ``2020-01-01_2021-01-01``

Region
~~~~~~

    The region is defined in the ``region_config`` inset. Its configuration must make a reference to the region spatial bins in ``region.txt``. It is also necessary to define a depth range, as well as a magnitude range and binning.

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
    The model configuration is set in the ``models`` inset with a list of model names, that specify their file paths and other attributes. In this case, we just reference the path to ``best_model.dat``, whose format is detected automatically.

    .. literalinclude:: ../../examples/case_a/config.yml
       :language: yaml
       :lines: 17-19

Evaluations
~~~~~~~~~~~
    The experiment's evaluations are defined in the ``tests`` inset. It consists of a list of evaluation names, making reference to its function and plotting function. These can be either defined in ``pycsep`` (see :doc:`pycsep:concepts/evaluations`) or manually. In this example, we employ the consistency N-test: its main function is found in :func:`csep.core.poisson_evaluations.number_test`, whereas its plotting function correspond to  :func:`csep.utils.plots.plot_poisson_consistency_test`

.. literalinclude:: ../../examples/case_a/config.yml
   :language: yaml
   :lines: 21-24


Running the experiment
----------------------

Run command
~~~~~~~~~~~

    The experiment can be run by simply navigating to the ``examples/case_a`` folder in the terminal an type.

    .. code-block:: shell

        fecsep run config.yml

    This will automatically set all the file paths of the calculation (testing catalogs, evaluation results, figures) and will display a summarized report in ``results/report.md``.

Plot command
~~~~~~~~~~~~

    If only plotting of the results is desired, when the results exist already, you can type:

    .. code-block:: shell

        fecsep plot config.yml

    This can be used, for example, when an additional plot is desired. Try adding to ``config.yml`` the following lines

    .. code-block:: yaml

        postproc_config:
          plot_forecasts: True

    and re-run with the ``plot`` command. A forecast figure will appear in ``results/2020-01-01_2021-01-01/forecasts``

Results
~~~~~~~

    The :func:`~fecsep.cmd.run` command creates automatically the results path tree for each time window analyzed. In the ``results/2020-01-01_2021-01-01/catalog`` folder, the testing catalog can be found (a subset of the global testing catalog) in ``json`` format.
    Human-readable results are found in ``results/2020-01-01_2021-01-01/evaluations``
    whereas their figures (and catalog) in ``results/2020-01-01_2021-01-01/figures``.
    Finally, the results are summarized in ``results/report.md``


Advanced
--------

The experiment run logic can be seen in the file ``case_a.py``, which executes the same example but in python source code. The run logic of the terminal commands ``run``, ``plot`` and ``reproduce`` can be seen in :class:`fecsep.cmd.main`


