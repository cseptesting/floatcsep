A - Simple(st) Time-Dependent, Catalog-based Model
==================================================

.. currentmodule:: floatcsep

.. contents::
    :local:

.. admonition:: **TL; DR**

    In a terminal, navigate to ``floatcsep/examples/case_f`` and type:

    .. code-block:: console

        $ floatcsep run config.yml

    After the calculation is complete, the results will be summarized in ``results/report.md``.


Artifacts
---------

This example shows how a time-dependent model should be set up for a time-dependent experiment
::

    case_f
        └──  pymock
            ├── input
                ├── args.txt  (model arguments)
                └── catalog.csv (dynamically allocated catalog)
            ├── pymock (model source code)
            └── forecasts (where forecasts will be stored)
        ├── catalog.csv
        ├── config.yml
        ├── models.yml
        └── tests.yml

* The model to be evaluated (``pymock``) is a source code that generates forecast for multiple time windows.

* The testing catalog `catalog.csv` works also as the input catalog, by being filtered until the starting test_date and allocated in `pymock/input` dynamically (before the model is run)


Model
-----

The experiment's complexity increases from time-independent to dependent, since we now need a **Model** (source code) that is able to generate forecast that changes every window. The model should be conceptualized as a **black-box**, whose only interface/interaction with the ``floatcsep`` system is to receive an input (i.e. input catalog) and generates an output (i.e. the forecasts).

* Input: The input consists in input **data** and **arguments**.

    1. The input data is, at the least, a catalog filtered until the forecast beginning, which is automatically allocated by ``fecsep`` in the `{model}/input` prior to each model's run. It is stored inside the model in ``csep.ascii`` format for simplicity's sake (see :doc:`pycsep:concepts/catalogs`).

    .. literalinclude:: ../../examples/case_f/catalog.csv
        :lines: 1-2

    2. The input arguments controls how the source code works. The minimum arguments to run a model (which should be modified dynamically during an experiment) are the forecast ``start_date`` and ``end_date``. The experiment will read `{model}/input/args.txt` and change the values of ``start_date = {datetime}`` and ``end_date = {datetime}`' before the model is run. Additional arguments can be set by convenience, such as ``catalog`` (the input catalog name), ``n_sims`` (number of synthetic catalogs) and random ``seed`` for reproducibility.

* Output: The model's output are the synthetic catalogs, which should be allocated in `{model}/forecasts/{filename}.csv`. The format is identically to ``csep_ascii``, but unlike in an input catalog, the ``catalog_id`` column should be modified for each synthetic catalog starting from 0. The file name follows the convention `{model_name}_{start}_{end}.csv`, where ``start`` and ``end`` folowws the `%Y-%m-%dT%H:%M:%S.%f` - ISO861 FORMAT

* Model run: The model should be run with a simple command to which only ``arguments`` should be passed. For this example, is

    .. code-block:: console

        $ python pymock/run.py input/args.txt

or using a binary (see ``pymock/setup.cfg:entry_point``)

    .. code-block:: console

        $ pymock input/args.txt




Configuration
-------------


Time
~~~~

    The configuration is identical to time-independent models, with the exception that now a ``horizon`` can be defined instead of ``intervals``, which is the forecast time-window length. The experiment's class should now be explicited as ``exp_class: td``

    .. literalinclude:: ../../examples/case_f/config.yml
       :language: yaml
       :lines: 3-7

Catalog
~~~~~~~

    The catalog was obtained ``previous to the experiment`` using ``query_bsi``, but it was filtered from 2006 onwards, so it has enough data for the model calibration.

Models
~~~~~~

    Additional arguments should be passed to time-independent models.

    .. literalinclude:: ../../examples/case_f/models.yml
       :language: yaml
       :lines: 3-7

    Now ``path`` points to the folder where the source is installed. Input and forecasts should be in ``{path}/input`` and ``{path}/forecasts``. The ``func`` option is the shell command with which the model is run (the shell will be run from the model's directory). Note that ``python run.py`` can be changed to ``pymock`` (see entry_point in ``pymock/setup.cfg``). In practice, the model will be run as

    .. code-block:: console

        $ python {path}/run.py {path}/input/args.txt

Tests
~~~~~

    With time-dependent models, now catalog evaluations found in :obj:`csep.core.catalog_evaluations` can be used.


    .. literalinclude:: ../../examples/case_f/tests.yml
       :language: yaml

    .. note::
        It is possible to assign two plotting functions to a test, whose ``plot_args`` and ``plot_kwargs`` can be placed indented beneath


Running the experiment
----------------------

    The experiment can be run by simply navigating to the ``examples/case_f`` folder in the terminal and typing.

    .. code-block:: console

       $ floatcsep run config.yml

    This will automatically set all the calculation paths (testing catalogs, evaluation results, figures) and will create a summarized report in ``results/report.md``.

