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
    The experiment region, catalog, forecasts and results can be viewed in the **Experiment Dashboard** with:

    .. code-block:: console

            $ floatcsep view config.yml


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

* The model to be evaluated (``etas``) is a collection of daily forecasts from ``2016-11-14`` until ``2016-11-21``. The forecasts are `Catalog-Based <https://docs.cseptesting.org/concepts/forecasts.html#catalog-based-forecasts>`_, which are composed of multiple individual simulations (See `Working with catalog-based forecasts <https://docs.cseptesting.org/concepts/forecasts.html#working-with-catalog-based-forecasts>`_)

.. important::
    The forecasts must be located in a folder ``forecasts`` inside the model folder. This is meant for consistency with models based on source codes (see subsequent tutorials).


Model
-----

    The time-dependency of a model is manifested here by the provision of different `Catalog-Based Forecasts <https://docs.cseptesting.org/concepts/forecasts.html#catalog-based-forecasts>`_, i.e., stochastic descriptions of seismicity, for different time-windows. In this example, the forecasts were created from an external `ETAS model <https://github.com/lmizrahi/etas>`_ (:ref:`Mizrahi et al. 2021 <case_f_references>`), with which the experiment has no interface for this case. This means that we use **only the forecast files** and no source code. We leave the handling of a model source code for tutorial :ref:`case_h`.



Configuration
-------------


Time
~~~~

    The configuration is analogous to time-independent models with multiple time-windows (e.g., :ref:`case_c`) with the exception that a ``horizon`` could be defined instead of ``intervals``, which is the forecast time-window length. The experiment's class should now be explicited as ``exp_class: td``.

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

    .. warning::
        For consistency with time-dependent models that will create forecasts from a source code, the ``path`` should point to the folder of the model, which itself should contain a sub-folder named ``{path}/forecasts`` where the files are located. For format descriptions, see `Working with catalog-based forecasts <https://docs.cseptesting.org/concepts/forecasts.html#working-with-catalog-based-forecasts>`_).

    .. important::
        Note that for catalog-based forecast models, the number of catalog simulations (``n_sims``) must be specified – because a forecast may contain synthetic catalogs with zero-event simulations and therefore does not imply the total number of simulated synthetic catalogs.


Tests
~~~~~

    Having a time-dependent and catalog-based forecast model, catalog-based evaluations found in :obj:`csep.core.catalog_evaluations` can now be used.


    .. literalinclude:: ../../tutorials/case_f/tests.yml
       :language: yaml

    .. note::
        It is possible to assign two plotting functions to a test, whose ``plot_args`` and ``plot_kwargs`` can be placed indented beneath.


.. note::

    For further details on how to configure an experiment, models and evaluations, see:

    - :ref:`experiment_config`
    - :ref:`model_config`
    - :ref:`evaluation_config`

Running the experiment
----------------------

    The experiment can be run by simply navigating to the ``tutorials/case_h`` folder in the terminal and typing:

    .. code-block:: console

       $ floatcsep run config.yml

    This will automatically set all the calculation paths (testing catalogs, evaluation results, figures) and will create a summarized report in ``results/report.md``.



pyCSEP under the hood
---------------------

    This tutorial uses *floatCSEP* as the orchestrator, but relies on *pyCSEP* for functions and objects.

    **Classes and functions used in this tutorial**

    - Catalog: :py:class:`csep.core.catalogs.CSEPCatalog`

        - :meth:`csep.core.catalogs.CSEPCatalog.load_json`
        - :meth:`csep.core.catalogs.CSEPCatalog.write_json`

    - Region: :py:class:`csep.core.regions.nz_csep_region`
    - Forecast class: :py:class:`csep.core.forecasts.CatalogForecast`

        - :meth:`csep.load_catalog_forecast`
        - :meth:`floatcsep.utils.file_io.CatalogForecastParsers.csv`

    - Test functions:

        - :py:func:`csep.core.catalog_evaluations.number_test`
        - :py:func:`csep.core.catalog_evaluations.spatial_test`

    - Result plotting functions:

        - :py:func:`csep.utils.plots.plot_number_test`
        - :py:func:`csep.utils.plots.plot_consistency_test`


    **Where to learn pyCSEP further:**

    - :doc:`pycsep:concepts/catalogs`
    - :doc:`pycsep:concepts/regions`
    - :doc:`pycsep:concepts/forecasts`
    - :doc:`pycsep:concepts/evaluations`


.. _case_f_references:

References
----------

    * Mizrahi, L., Nandan, S., & Wiemer, S. (2021). The effect of declustering on the size distribution of mainshocks. _Seismological Research Letters, 92_(4), 2333–2342. doi: `10.1785/0220200231 <https://doi.org/10.1785/0220200231>`_