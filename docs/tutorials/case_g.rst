.. _case_g:

G - Testing a Time-Dependent Model
==================================

Here, we set up a time-dependent model from its **source code** for an experiment.

.. admonition:: **TL; DR**

    In a terminal, navigate to ``floatcsep/tutorials/case_g`` and type:

    .. code-block:: console

        $ floatcsep run config.yml

    After the calculation is complete, the results will be summarized in ``results/report.md``.


.. currentmodule:: floatcsep

.. contents:: Contents
    :local:



Experiment Components
---------------------

The example folder contains also, along with the already known components (configurations, catalog), a sub-folder for the **source code** of the model  ``pymock``. The components of the experiment (and model) are:

::

    case_g
        └──  pymock         (Model's source code)
            ├── input           (input interface to floatcsep)
                ├── args.txt        (model arguments)
                └── catalog.csv     (dynamically allocated catalog)
            ├── pymock
                ├── libs.py         (helper functions)
                └── main.py         (main routines)
            └── forecasts       (output interface to floatcsep)
                ... (forecasts should be stored here when the model is run)
            ├── run.py          (One of the possibilities to run the model)
            ├── pyproject.toml  (Build instructions)
            ├── setup.cfg       (Build instructions)
            ├── setup.py        (Build instructions)
            ├── requirements.txt(Build instructions)
            ├── Dockerfile      (Build instructions)
            └── README.md       (Information)

        ├── catalog.csv
        ├── config.yml
        ├── models.yml
        ├── custom_plot_script.py
        └── tests.yml

* The model to be evaluated (``pymock``) is a source code that generates forecasts for multiple time windows.

* The testing catalog ``catalog.csv`` works also as the input catalog, by being filtered until the testing ``start_date`` and allocated in `pymock/input` dynamically (before each time the model is run)

.. _Model:

Model
-----

The experiment's complexity increases from time-independent to dependent mostly because we now need a **Model** (source code) to generate forecasts that changes for every time-window. The model main components are:


* **Input**: The input consists in input **data** and **arguments**.

    1. The **input data** is, at the very least, a catalog filtered until the forecast beginning. The catalog will be automatically allocated by ``floatcsep`` prior to each model's run (e.g., a single forecast run) in the `{model}/input` folder. It is stored in the ``csep.ascii`` format for simplicity's sake (see :doc:`pycsep:concepts/catalogs`).

    .. literalinclude:: ../../tutorials/case_g/catalog.csv
        :caption: tutorials/case_g/catalog.csv
        :lines: 1-2

    2. The **input arguments** controls how the model's source code works. The minimum arguments to run a model are the forecast ``start_date`` and ``end_date``, which will be modified dynamically during an experiment with multiple time-windows. The experiment system will access `{model}/input/args.txt` and change the values of ``start_date = {datetime}`` and ``end_date = {datetime}`` before the model is run. Additional arguments can be set by convenience, such as (not limited to) ``catalog`` (the input catalog name), ``n_sims`` (number of synthetic catalogs) and random ``seed`` for reproducibility.

* **Output**: The model's output are the synthetic catalogs, which should be allocated in `{model}/forecasts/{filename}.csv` by the source code after each rone. The format is identically to ``csep_ascii``, but unlike in an input catalog, the ``catalog_id`` column should be modified for each synthetic catalog starting from 0. The file name follows the convention `{model_name}_{start}_{end}.csv`, where ``start`` and ``end`` folows the `%Y-%m-%dT%H:%M:%S.%f` - ISO861 FORMAT

* **Model build**: Inside the model source code, there are multiple options to build it. A standard python ``setup.cfg`` is given, which can be built inside a python ``venv`` or ``conda`` managers. This is created and built automatically by ``floatCSEP``, as long as the the model build instructions are correctly set up.

* **Model run**: The model should be run with a simple command, e.g. **entrypoint**, to which only ``arguments`` could be passed if desired. The ``pymock`` model contains multiple example of entrypoints, but the modeler should use only one for clarity.

    1.  A `python` call with arguments

    .. code-block:: console

        $ python run.py input/args.txt

    2. Using a binary entrypoint with arguments (for instance, defined in the python build instructions: ``pymock/setup.cfg:entry_point``)

    .. code-block:: console

        $ pymock input/args.txt

    3. A single binary entrypoint without arguments .

    .. code-block:: console

        $ pymock

    This means that the source code should internally read the input data and arguments, ``input/catalog.csv`` and  ``input/args.txt`` files respectively.

.. important::

    The model should be conceptualized as a **black-box**, whose only interface/interaction with the ``floatcsep`` system is to receive an input (i.e., input catalog and arguments) and generates an output (the forecasts).


Configuration
-------------


Time
~~~~

    The configuration is identical to time-independent models, with the exception that now a ``horizon`` can be defined instead of ``intervals``, which is the forecast time-window length. The experiment's class should now be explicited as ``exp_class: td``

    .. literalinclude:: ../../tutorials/case_g/config.yml
        :caption: tutorials/case_g/config.yml
        :language: yaml
        :lines: 3-7

Catalog
~~~~~~~

    The catalog was obtained `previous to the experiment` using ``query_bsi``, but it was filtered from 2006 onwards, so it has enough data for the model calibration.

Models
~~~~~~

    Additional arguments should be passed to time-independent models.

    .. literalinclude:: ../../tutorials/case_g/models.yml
        :caption: tutorials/case_g/models.yml
        :language: yaml
        :lines: 1-7

    1. Now ``path`` points to the folder where the source is installed. Therefore, the input and the forecasts should be allocated ``{path}/input`` and ``{path}/forecasts``, respectively.
    2. The ``func`` option is the shell command with which the model is run. As seen in the `Model`_ section, this could be either ``pymock``, ``pymock input/args.txt`` or ``python run.py input/args``. We use the simplest option ``pymock``, but you are welcome to try different entrypoints.

    .. note::
        The ``func`` command will be run from the model's directory and a model containerization (e.g., ``Dockerfile``, ``conda``).

    3. The ``func_kwargs`` are extra arguments that will annotated to the ``input/args.txt`` file every time the model is run, or will be passed as extra arguments to the ``func`` call (Note that the two options are identical). This is useful to define sub-classes of models (or flavours) that uses the same source code, but a different instantiation.
    4. The ``build`` option defines the style of container within which the model will be placed. Currently in **floatCSEP**, only the python module ``venv``, the package manager ``conda`` and the containerization manager ``Docker`` are currently supported.

    .. important::
        For these tutorials, we use ``venv`` sub-environments, but we recommend ``Docker`` to set up real experiments.


Tests
~~~~~

    With time-dependent models, now catalog evaluations found in :obj:`csep.core.catalog_evaluations` can be used.


    .. literalinclude:: ../../tutorials/case_g/tests.yml
        :caption: tutorials/case_g/tests.yml
        :language: yaml

    .. note::
        It is possible to assign two plotting functions to a test, whose ``plot_args`` and ``plot_kwargs`` can be placed indented beneath


Custom Post-Process
~~~~~~~~~~~~~~~~~~~

    Additional to the default :func:`~floatcsep.postprocess.plot_handler.plot_results`, :func:`~floatcsep.postprocess.plot_handler.plot_catalogs`, :func:`~floatcsep.postprocess.plot_handler.plot_forecasts` functions, a custom plotting function(s) can be set within the ``postprocess`` configuration

    .. literalinclude:: ../../tutorials/case_g/config.yml
        :caption: tutorials/case_g/config.yml
        :language: yaml
        :lines: 22-23

    This option provides `hook` for a python script and a function within as:

    .. code-block:: console

        {python_sript}:{function_name}

    The requirements are that the script to be located within the same directory as the configuration file, whereas the function must receive a :class:`floatcsep.experiment.Experiment` as argument

    .. literalinclude:: ../../tutorials/case_g/custom_plot_script.py
        :caption: tutorials/case_g/custom_plot_script.py
        :language: python
        :lines: 6-13



    In this way, the plot function can use all the :class:`~floatcsep.experiment.Experiment` attributes/methods to access catalogs, forecasts and test results. The script ``tutorials/case_g/custom_plot_script.py`` can also be viewed directly on `GitHub <https://github.com/cseptesting/floatcsep/blob/main/tutorials/case_g/custom_plot_script.py>`_, where it is exemplified how to access the experiment data in runtime.


Running the experiment
----------------------

    The experiment can be run by simply navigating to the ``tutorials/case_g`` folder in the terminal and typing.

    .. code-block:: console

       $ floatcsep run config.yml

    This will automatically set all the calculation paths (testing catalogs, evaluation results, figures) and will create a summarized report in ``results/report.md``.

