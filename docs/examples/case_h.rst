H - Multiple Catalog-Based Models
=================================

.. currentmodule:: floatcsep

.. contents::
    :local:
    :depth: 1

.. admonition:: **TL; DR**

    In a terminal, navigate to ``floatcsep/examples/case_h`` and type:

    .. code-block:: console

        $ floatcsep run config.yml

    After the calculation is complete, the results will be summarized in ``results/report.md``.


Experiment Components
---------------------

This example shows how to run an experiment that access, downloads, containerize and execute multiple time-dependent models. The experiment input files are:

::

    case_h
        ├── catalog.csv
        ├── config.yml
        ├── models.yml
        └── tests.yml

* The ``models.yml`` contains the instructions to clone and build the source codes from software repositories (e.g., gitlab, Github)


Models
------

As in Example G, the complexity increases because each **Model** requires to build and execute a source code to generate forecast for multilpe time-windows. The instructions for each model are located within ``models.yml``. The URL and specific version (e.g., commit hash, tag, release) are specified as:


    .. literalinclude:: ../../examples/case_h/models.yml
        :lines: 1-3

The model source code requires to be synchronized with the experiment interface, in such a way that is able to read the inputs (catalog and argument file) from an ``{model_path}/input`` folder and store the forecast outputs from a folder ``{model_path}/forecasts``.

    .. literalinclude:: ../../examples/case_h/models.yml
        :lines: 4-6

The ``func`` parameter indicates how the model should be invoked from a terminal (e.g, a python virtual environment, docker container, etc.). The type of container is set with the parameter ``build`` (``floatcsep`` supports ``conda``, ``venv`` and ``Dockerfile``). Note that we specify to ``floatcsep`` which arguments file will be read from the module. In this case, the ETAS model uses ``args.json`` file (``floatcsep`` suports ``.json`` and ``.txt`` files). ``floatcsep`` will dynamically modify the ``start_time`` and ``end_time`` for each time window run, and statically (i.e., for all time-windows) allocate the parameters set as:

    .. literalinclude:: ../../examples/case_h/models.yml
        :lines: 7-9

Same as Example G, the repositories should follow the following:


* **Input**: The input consists in input **data** and **arguments**.

    1. The **input data** is, at the least, a catalog filtered until the forecast beginning, which is automatically allocated by ``floatcsep`` in the `{model}/input` prior to each model's run. It is stored inside the model in ``csep.ascii`` format for simplicity's sake (see :doc:`pycsep:concepts/catalogs`).

    .. literalinclude:: ../../examples/case_h/catalog.csv
        :lines: 1-2

    2. The **input arguments** controls how the model's source code works. The minimum arguments to run a model (which should be modified dynamically during an experiment) are the forecast ``start_date`` and ``end_date``. The experiment will access `{model}/input/args.json` and change the values of ``"start_date": "{datetime}"`` and ``"end_date": "{datetime}"`` before the model is run. Additional arguments can be set by convenience, such as ``n_sims`` (number of synthetic catalogs), ``m_c`` for the cutoff magnitude and a random ``seed`` for reproducibility.

* **Output**: The model's output are the synthetic catalogs, which should be allocated in `{model}/forecasts/{filename}.csv`. The format is identically to ``csep_ascii``, but unlike in an input catalog, the ``catalog_id`` column should be modified for each synthetic catalog starting from 0. The file name follows the convention `{model_name}_{start}_{end}.csv`, where ``start`` and ``end`` folowws the `%Y-%m-%dT%H:%M:%S.%f` - ISO861 FORMAT

* **Model build**: Inside the model source code, there are multiple options to build it. A standard python ``setup.cfg`` is given, which can be built inside a python ``venv`` or ``conda`` managers. This is created and built automatically by ``floatCSEP``, as long as the the model build instructions are correctly set up.

* **Model run**: The model should be run with a simple command to which only ``arguments`` should be passed. For this example, is

    .. code-block:: console

        $ etas-run


    as long as it internally reads the ``input/args.json`` and ``input/catalog.csv`` files.


Configuration
-------------


Time
~~~~

    The configuration is identical to time-independent models, with the exception that now a ``horizon`` can be defined instead of ``intervals``, which is the forecast time-window length. The experiment's class should now be explicited as ``exp_class: td``

    .. literalinclude:: ../../examples/case_h/config.yml
       :language: yaml
       :lines: 3-7

Catalog
~~~~~~~

    The catalog was obtained ``previous to the experiment`` using ``query_bsi``, but it was filtered from 2006 onwards, so it has enough data for the model calibration.


Tests
~~~~~

    With time-dependent models, now catalog evaluations found in :obj:`csep.core.catalog_evaluations` can be used.


    .. literalinclude:: ../../examples/case_h/tests.yml
       :language: yaml

    .. note::
        It is possible to assign two plotting functions to a test, whose ``plot_args`` and ``plot_kwargs`` can be placed indented beneath


Custom Post-Process
~~~~~~~~~~~~~~~~~~~

    A custom reporting function can be set within the ``postprocess`` configuration to replace the :func:`~floatcsep.postprocess.reporting.generate_report`:

    .. literalinclude:: ../../examples/case_h/config.yml
       :language: yaml
       :lines: 22-23

    This option provides `hook` for a python script and a function within as:

    .. code-block:: console

        {python_sript}:{function_name}

    The requirements are that the script to be located within the same directory as the configuration file, whereas the function must receive a :class:`floatcsep.experiment.Experiment` as argument

    .. literalinclude:: ../../examples/case_h/custom_report.py
       :language: yaml
       :lines: 5-11

    In this way, the report function use all the :class:`~floatcsep.experiment.Experiment` attributes/methods to access catalogs, forecasts and test results. The script ``examples/case_h/custom_report.py`` can also be viewed directly on `GitHub <https://github.com/cseptesting/floatcsep/blob/main/examples/case_h/custom_report.py>`_, where it is exemplified how to access the experiment artifacts.


Running the experiment
----------------------

    The experiment can be run by simply navigating to the ``examples/case_h`` folder in the terminal and typing.

    .. code-block:: console

       $ floatcsep run config.yml

    This will automatically set all the calculation paths (testing catalogs, evaluation results, figures) and will create a summarized report in ``results/report.md``.

