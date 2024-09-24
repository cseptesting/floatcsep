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
    In this experiment no region file is needed, because the region is encoded in the forecasts themselves (QuadTree models, see https://zenodo.org/record/6289795 and https://zenodo.org/record/6255575 ).

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



