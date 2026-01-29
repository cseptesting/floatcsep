.. _case_d:

D - Catalog and Model Queries
=============================

The following example shows an experiment whose forecasts are **retrieved from a repository** (Zenodo - https://zenodo.org) and the testing **catalog** from an authoritative source **web service** (namely the gCMT catalog from the International Seismological Centre - http://www.isc.ac.uk).



.. currentmodule:: floatcsep

.. admonition:: **TL; DR**

    In a terminal, navigate to ``floatcsep/tutorials/case_d`` and type:

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

The source code can be found in the ``tutorials/case_d`` folder or in  `GitHub <https://github.com/cseptesting/floatcsep/blob/main/tutorials/case_d>`_. The **initial** input structure of the experiment is:

::

    case_d
        ├── config.yml
        ├── models.yml
        └── tests.yml

Once the catalog and models have been downloaded, the experiment structure will look like this:

::

    case_d
        └──  models
            └──  team
                ├── TEAM=N10L11.csv
                ├── TEAM=N25L11.csv
                ...
            └──  wheel
                ├── WHEEL=N10L11.csv
                ├── WHEEL=N25L11.csv
                ...
        ├── config.yml
        ├── catalog.json
        ├── models.yml
        └── tests.yml

.. note::
    In this experiment no region file is needed because the region is encoded in the forecasts themselves, which are based on the QuadTree description (See `Working with quadtree-gridded forecasts <https://docs.cseptesting.org/concepts/forecasts.html#working-with-quadtree-gridded-forecasts>`_, and the Zenodo repositories https://zenodo.org/record/6289795 and https://zenodo.org/record/6255575 ).

Configuration
-------------

Catalog
~~~~~~~

    The ``catalog`` inset from ``config.yml`` now makes reference to a catalog query function, in this case :func:`~pycsep.query_gcmt`.

    .. literalinclude:: ../../tutorials/case_d/config.yml
        :caption: tutorials/case_d/config.yml
        :language: yaml
        :lines: 14-14

    ``floatcsep`` will automatically filter the catalog to the experiment time, spatial and magnitude windows of the experiment.

    .. note::

     Query functions are located in ``pycsep`` (e.g. :func:`csep.query_comcat`, :func:`csep.query_bsi`, :func:`csep.query_gcmt`, :func:`csep.query_gns`). Only the name of the function is needed to retrieve the catalog. Refer to :obj:`csep` API reference.

Models
~~~~~~
    The model configuration is set in ``models.yml``.

    .. literalinclude:: ../../tutorials/case_d/models.yml
        :caption: tutorials/case_d/models.yml
        :language: yaml

    * The option ``zenodo_id`` makes reference to the zenodo **record id**. The model ``team`` is found in https://zenodo.org/record/6289795, whereas the model ``wheel`` in https://zenodo.org/record/6255575.

    * The ``zenodo`` (or ``git``) repositories could contain multiple files, each of which can be assigned to a flavour.

    * The option ``flavours`` allows multiple model sub-classes to be quickly instantiated.

    * When multiple flavours are passed, ``path`` refers to the folder where the models would be downloaded.

    * If a single file of the repository is needed (without specifying model flavours), ``path`` can reference to the file itself. For example, you can try replacing the whole WHEEL inset in ``models.yml`` to:

        .. code-block:: yaml

            - WHEEL:
                zenodo_id: 6255575
                path: models/WHEEL=N10L11.csv


Running the experiment
----------------------

    The experiment can be run by simply navigating to the ``tutorials/case_d`` folder in the terminal and typing.

    .. code-block:: console

        $ floatcsep run config.yml

    This will automatically set all the calculation paths (testing catalogs, evaluation results, figures) and will create a summarized report in ``results/report.md``.



pyCSEP under the hood
---------------------

    This tutorial uses *floatCSEP* as the orchestrator, but relies on *pyCSEP* for functions and objects.

    **Classes and functions used in this tutorial**

    - Catalog: :py:class:`csep.core.catalogs.CSEPCatalog`

        - :func:`csep.load_catalog`
        - :meth:`csep.core.catalogs.CSEPCatalog.write_json`

    - Region: :py:class:`csep.core.regions.QuadtreeGrid2D`
    - Forecast class: :py:class:`csep.core.forecasts.GriddedForecast`

        - :meth:`floatcsep.utils.file_io.GriddedForecastParsers.quadtree`

    - Test functions:

        - :py:func:`csep.core.poisson_evaluations.spatial_test`
        - :py:func:`csep.core.poisson_evaluations.paired_t_test`
        - :py:func:`floatcsep.utils.helpers.vector_poisson_t_w_test`

    - Result plotting functions:

        - :py:func:`csep.utils.plots.plot_poisson_consistency_test`
        - :py:func:`csep.utils.plots.plot_comparison_test`
        - :py:func:`floatcsep.utils.helpers.plot_matrix_comparative_test`

    **Where to learn pyCSEP further:**

    - :doc:`pycsep:concepts/catalogs`
    - :doc:`pycsep:concepts/regions`
    - :doc:`pycsep:concepts/forecasts`
    - :doc:`pycsep:concepts/evaluations`
