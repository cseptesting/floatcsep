.. _case_h:

H - A Time-Dependent Experiment
===============================

Here, we run an experiment that accesses, containerizes and executes multiple **time-dependent models**, and then proceeds to evaluate the forecasts once they are created.

.. admonition:: **TL; DR**

    In a terminal, navigate to ``floatcsep/tutorials/case_h`` and type:

    .. code-block:: console

        $ floatcsep run config.yml

    After the calculation is complete, the results will be summarized in ``results/report.md``.

.. currentmodule:: floatcsep

.. contents:: Contents
    :local:



Experiment Components
---------------------

The experiment input files are:

::

    case_h
        ├── catalog.csv
        ├── config.yml
        ├── tests.yml
        └── models.yml

* The ``models.yml`` contains the instructions to clone and build the source codes from software repositories (e.g., gitlab, Github), and how to interface them with **floatCSEP**. Once downloaded and built, the experiment structure should look like this:

::

    case_h
        ├── models
            ├── etas
                └── ...     (ETAS source code)
            ├── pymock_poisson
                └── ...     (Poisson-PyMock source code)
            └── pymock_nb
                └── ...     (Negative-Binomial-PyMock source code)
        ├── catalog.csv
        ├── config.yml
        ├── tests.yml
        ├── custom_report.py
        └── models.yml

Configuration
-------------

Models
~~~~~~

As in :ref:`Tutorial G<case_g>`, each **Model** requires to build and execute a source code to generate forecasts. The instructions for each model are located within ``models.yml``, which we further explain here:

.. note::
    The ``models.yml`` will define how to interface **floatCSEP** to each Model, implying that a Model should be developed, or adapted to ensure the interface requirements specified below.

1. The repository URL of each model and their specific versions (e.g., commit hash, tag, release) are specified as:

    .. literalinclude:: ../../tutorials/case_h/models.yml
        :caption: tutorials/case_h/models.yml
        :language: yaml
        :lines: 1-3, 11-13, 21-23

2. A ``path`` needs to be indicated for each model, to both download the repository contents therein and from where the source code will be executed.

    .. literalinclude:: ../../tutorials/case_h/models.yml
        :caption: tutorials/case_h/models.yml
        :language: yaml
        :lines: 1-4
        :emphasize-lines: 4
        :lineno-match:

    .. important::
     The implies that the inputs (catalog and argument file) should be read by the model from a ``{path}/input`` folder, and the forecast outputs stored into a folder ``{path}/forecasts``.

2. There is some flexibility to interface **floatCSEP** with a model. For instance, a different `filepath` can be set for the argument file:

    .. literalinclude:: ../../tutorials/case_h/models.yml
        :caption: tutorials/case_h/models.yml
        :language: yaml
        :lines: 5
        :lineno-match:

    .. note::
        **floatCSEP** can read `.txt`, `.json` and `.yml` format-styled argument files. In all cases, the minimum required arguments, should be formatted as:

        .. code-block:: console

            #.txt
            start_date = {DATESTRING}
            end_date = {DATESTRING}

        .. code-block:: yaml

            #.yml
            start_date: {DATESTRING}
            end_date: {DATESTRING}

        .. code-block:: json

            //.json
            "start_date": "{DATESTRING}",
            "end_date": "{DATESTRING}"

        **floatcsep** will dynamically modify the ``start_date`` and ``start_date`` for each time window run, and any extra arguments will just be added for all time-windows.

4. The ``func`` entry indicates how the models are invoked from a shell terminal.

    .. literalinclude:: ../../tutorials/case_h/models.yml
        :caption: tutorials/case_h/models.yml
        :language: yaml
        :lines: 1,6,11,15,21,25

    .. important::
        Please refer to :ref:`Tutorial G<case_g>` for example of how to set up ``func`` for the model and interface it to **floatCSEP**.

5. A prefix for the resulting forecast filepaths can be specified beforehand for each model. Without specifying this, the default is ``{model_name}`` (e.g., `etas`, `Poisson Mock`, `Negbinom Mock`).

    .. literalinclude:: ../../tutorials/case_h/models.yml
        :caption: tutorials/case_h/models.yml
        :language: yaml
        :lines: 21, 26

    The experiment will read the forecasts as:

    .. code-block::

        {model_path}/{forecasts}/{prefix}_{start}_{end}.csv

    where ``start`` and ``end`` follow either the ``%Y-%m-%dT%H:%M:%S.%f`` - ISO861 FORMAT, or the short date version ``%Y-%m-%d`` if the windows are set by midnight.

6. Additional function arguments can be passed to the model with the entry ``func_kwargs``. Both `Poisson Mock` and `Negbinom Mock` use the same source code, but a different subclass can be defined with ``func_kwargs`` (in this case, a Negative-Binomial number distribution instead of Poisson).

    .. literalinclude:: ../../tutorials/case_h/models.yml
        :caption: tutorials/case_h/models.yml
        :language: yaml
        :lines: 11,17-20,21,27-31


Time
~~~~

    The configuration is identical to time-independent models, with the exception that now a ``horizon`` can be defined instead of ``intervals``, which is the forecast time-window length. The experiment's class should now be explicited as ``exp_class: td``

    .. literalinclude:: ../../tutorials/case_h/config.yml
        :caption: tutorials/case_h/config.yml
        :language: yaml
        :lines: 3-7

Catalog
~~~~~~~

    The catalog was obtained *prior* to the experiment using ``query_bsi``, but it was filtered from 2006 onwards, so it has enough data for the model calibration.


Tests
~~~~~

    Catalog-based evaluations found in :obj:`csep.core.catalog_evaluations` can be used.


    .. literalinclude:: ../../tutorials/case_h/tests.yml
        :caption: tutorials/case_h/tests.yml
        :language: yaml

    .. note::
        It is possible to assign two plotting functions to a test, whose ``plot_args`` and ``plot_kwargs`` can be placed indented beneath.


Custom Post-Process
~~~~~~~~~~~~~~~~~~~

    A custom reporting function can be set within the ``postprocess`` configuration to replace the :func:`~floatcsep.postprocess.reporting.generate_report`:

    .. literalinclude:: ../../tutorials/case_h/config.yml
        :caption: tutorials/case_h/config.yml
        :language: yaml
        :lines: 22-23

    This option provides `hook` for a Python script and a function within as:

    .. code-block:: console

        {python_sript}:{function_name}

    The script must be located within the same directory as the configuration file, whereas the function must receive a :class:`floatcsep.experiment.Experiment` as argument:

    .. literalinclude:: ../../tutorials/case_h/custom_report.py
       :language: yaml
       :lines: 5-11

    In this way, the report function use all the :class:`~floatcsep.experiment.Experiment` attributes/methods to access catalogs, forecasts and test results. The script ``tutorials/case_h/custom_report.py`` can also be viewed directly in `the GitHub repository <https://github.com/cseptesting/floatcsep/blob/main/tutorials/case_h/custom_report.py>`_, where it is exemplified how to access the experiment artifacts.


Running the experiment
----------------------

    The experiment can be run by simply navigating to the ``tutorials/case_h`` folder in the terminal and typing:

    .. code-block:: console

       $ floatcsep run config.yml

    This will automatically set all the calculation paths (testing catalogs, evaluation results, figures) and will create a summarized report in ``results/report.md``.

