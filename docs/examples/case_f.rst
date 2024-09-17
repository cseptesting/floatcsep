G - Time-Dependent, Catalog-Based Model (from existing files)
==========================================================

.. currentmodule:: floatcsep

.. contents::
    :local:

.. admonition:: **TL; DR**

    In a terminal, navigate to ``floatcsep/examples/case_f`` and type:

    .. code-block:: console

        $ floatcsep run config.yml

    After the calculation is complete, the results will be summarized in ``results/report.md``.


Experiment Components
---------------------


This example shows how set up a time-dependent model, whose files are already existing, and experiment. The model structure is as follows:
::

    case_f
        └──  etas
            ├── forecasts
                ├── etas_2016-11-14_2016-11-15.csv  (forecast files)
                ...
                └── etas_2016-11-20_2016-11-21.csv
        ├── catalog.csv
        ├── config.yml
        ├── models.yml
        └── tests.yml

* The model to be evaluated (``etas``) is a collection of daily forecasts from ``2016-11-14`` until ``2016-11-21``.

* The forecasts are located in a folder ``forecasts`` inside the model, to be consistent with models based on source codes (see the subsequent examples).


Model
-----

The time-dependency of a model is manifested here by the provision of different forecasts, i.e., statistical descriptions of seismicity, for different time-windows. For this example, the forecasts were created from an external model https://github.com/lmizrahi/etas (https://doi.org/10.1785/0220200231
), with which the experiment has no interface. This means that we only the forecast files are required. We leave the handling of a model source code for subsequent examples.



Configuration
-------------


Time
~~~~

    The configuration is identical to time-independent models with multiple time-windows (e.g., case C), with the exception that now a ``horizon`` can be defined instead of ``intervals``, which is the forecast time-window length. The experiment's class should now be explicited as ``exp_class: td``

    .. literalinclude:: ../../examples/case_f/config.yml
       :language: yaml
       :lines: 3-7

Catalog
~~~~~~~

    The catalog was obtained ``previous to the experiment`` using ``query_geonet`` and it was filtered to the testing period.

Models
~~~~~~

    Some additional arguments should be passed to a time-dependent model

    .. literalinclude:: ../../examples/case_f/models.yml
       :language: yaml
       :lines: 1-4

    For consistency with time-dependent models that will create forecasts from a source code, the ``path`` should point to the folder of the model. The path folder should contain a sub-folder named ``{path}/forecasts`` where the files are located. Note that fore catalog-based forecasts, the number of simulations should be explicit.

Tests
~~~~~

    With time-dependent models, now catalog evaluations found in :obj:`csep.core.catalog_evaluations` can be used.


    .. literalinclude:: ../../examples/case_f/tests.yml
       :language: yaml

    .. note::
        It is possible to assign two plotting functions to a test, whose ``plot_args`` and ``plot_kwargs`` can be placed indented beneath


Running the experiment
----------------------

    The experiment can be run by simply navigating to the ``examples/case_h`` folder in the terminal and typing.

    .. code-block:: console

       $ floatcsep run config.yml

    This will automatically set all the calculation paths (testing catalogs, evaluation results, figures) and will create a summarized report in ``results/report.md``.

