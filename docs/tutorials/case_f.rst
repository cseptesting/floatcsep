.. _case_f:

F - Testing Catalog-Based Forecasts
===================================

This example shows how set up an experiment with a **time-dependent** model, whose forecast files already exist.

.. currentmodule:: floatcsep

.. admonition:: **TL; DR**

    In a terminal, navigate to ``floatcsep/tutorials/case_f`` and type:

    .. code-block:: console

        $ floatcsep run config.yml

    After the calculation is complete, the results will be summarized in ``results/report.md``.

.. contents:: Contents
    :local:


Experiment Components
---------------------


The source files can be found in the ``tutorials/case_e`` folder or in  `the GitHub repository <https://github.com/cseptesting/floatcsep/blob/main/tutorials/case_e>`_. The experiment structure is as follows:

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

.. important::
    The forecasts must be located in a folder ``forecasts`` inside the model folder. This is meant for consistency with models based on source codes (see subsequent tutorials).


Model
-----

The time-dependency of a model is manifested here by the provision of different forecasts, i.e., statistical descriptions of seismicity, for different time-windows. In this example, the forecasts were created from an external model https://github.com/lmizrahi/etas (:ref:`Mizrahi et al. 2021<References>`_), with which the experiment has no interface. This means that we use **only the forecast files** and no source code. We leave the handling of a model source code for subsequent tutorials.



Configuration
-------------


Time
~~~~

    The configuration is analogous to time-independent models with multiple time-windows (e.g., case C) with the exception that a ``horizon`` could be defined instead of ``intervals``, which is the forecast time-window length. The experiment's class should now be explicited as ``exp_class: td``.

    .. literalinclude:: ../../tutorials/case_f/config.yml
        :caption: tutorials/case_f/config.yml
        :language: yaml
        :lines: 3-7

.. note::
    **floatCSEP** is flexible with the definition of time windows/deltas. Alternative string inputs for ``horizon`` can be ``1-day``, ``1 day``, ``1d``, etc.

Catalog
~~~~~~~

    The catalog ``catalog.json`` was obtained *prior* to the experiment by using ``query_geonet`` and it was filtered to the testing period. However, it can be re-queried by changing its definition to:

    .. code-block:: yaml

          catalog: query_geonet

Models
~~~~~~

    Some additional arguments should be passed to a **time-dependent** model, such as its class ('td' for time-dependent) and the number of simulations.

    .. literalinclude:: ../../tutorials/case_f/models.yml
        :caption: tutorials/case_f/config.yml
        :language: yaml
        :lines: 1-4

.. note::
    For consistency with time-dependent models that will create forecasts from a source code, the ``path`` should point to the folder of the model, which itself should contain a sub-folder named ``{path}/forecasts`` where the files are located.

.. important::
    Note that for catalog-based forecast models, the number of catalog simulations (``n_sims``) must be specified – because a forecast may contain synthetic catalogs with zero-event simulations and therefore does not imply the total number of simulated synthetic catalogs.

Tests
~~~~~

    Having a time-dependent and catalog-based forecast model, catalog-based evaluations found in :obj:`csep.core.catalog_evaluations` can now be used.


    .. literalinclude:: ../../tutorials/case_f/tests.yml
       :language: yaml

    .. note::
        It is possible to assign two plotting functions to a test, whose ``plot_args`` and ``plot_kwargs`` can be placed indented beneath.


Running the experiment
----------------------

    The experiment can be run by simply navigating to the ``tutorials/case_h`` folder in the terminal and typing:

    .. code-block:: console

       $ floatcsep run config.yml

    This will automatically set all the calculation paths (testing catalogs, evaluation results, figures) and will create a summarized report in ``results/report.md``.


References
----------

    * Mizrahi, L., Nandan, S., & Wiemer, S. (2021). The effect of declustering on the size distribution of mainshocks. _Seismological Research Letters, 92_(4), 2333–2342. doi: `10.1785/0220200231 <https://doi.org/10.1785/0220200231>`_