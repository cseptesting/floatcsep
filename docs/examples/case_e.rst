E - A Realistic Time-independent Experiment
===========================================

.. currentmodule:: floatcsep

.. contents::
    :local:

.. admonition:: **TL; DR**

    In a terminal, navigate to ``floatcsep/examples/case_e`` and type:

    .. code-block:: console

        $ floatcsep run config.yml

    After the calculation is complete, the results will be summarized in ``results/report.md``.


Artifacts
---------

This example shows how to run a realistic testing experiment in Italy (based on https://doi.org/10.4401/ag-4844), summarizing the concepts of previous examples. The example has only a subset of the original models and evaluations. The input structure of the experiment is:

::

    case_e
        └──  models
            ├── gulia-wiemer.ALM.italy.10yr.2010-01-01.xml
            ├── meletti.MPS04.italy.10yr.2010-01-01.xml
            └── zechar.TripleS-CPTI.italy.10yr.2010-01-01.xml
        ├── config.yml
        ├── models.yml
        └── tests.yml


Configuration
-------------


Time
~~~~

    The time configuration is manifested in the ``time-config`` inset.

    .. literalinclude:: ../../examples/case_e/config.yml
       :language: yaml
       :lines: 3-7

Region
~~~~~~

    The testing region is the official Italy CSEP Region obtained from :obj:`csep.core.regions.italy_csep_region`.

    .. literalinclude:: ../../examples/case_e/config.yml
       :language: yaml
       :lines: 9-15


Catalog
~~~~~~~

    The catalog is obtained from an authoritative source, namely the Bollettino Sismico Italiano (http://terremoti.ingv.it/en/bsi ), using the function :func:`csep.query_bsi`

    .. literalinclude:: ../../examples/case_e/config.yml
       :language: yaml
       :lines: 17-17

Models
~~~~~~
    The models are set in ``models.yml``. For simplicity, there are only three of the nineteen models originally submitted to the Italy Experiment.

    .. literalinclude:: ../../examples/case_e/models.yml
       :language: yaml

    The ``.xml`` format is automatically detected and parsed by ``floatcsep`` readers.

    .. note::

        The forecasts are defined in ``[Earthquakes / 10-years]``, specified with the ``forecast_unit`` option.

    .. note::

        The ``use_db`` flag allows ``floatcsep`` to transform the forecasts into a database (HDF5), which speeds up the calculations.

Post-Process
~~~~~~~~~~~~

    Additional options for post-processing can set using the ``postproc_config`` option.

    .. literalinclude:: ../../examples/case_e/config.yml
       :language: yaml
       :lines: 21-34

    See :func:`~csep.utils.plots.plot_spatial_dataset` for forecast plot options and :func:`~csep.utils.plots.plot_catalog` for the catalog placed on top.


Running the experiment
----------------------

    The experiment can be run by simply navigating to the ``examples/case_a`` folder in the terminal and typing.

    .. code-block:: console

        floatcsep run config.yml

    This will automatically set all the calculation paths (testing catalogs, evaluation results, figures) and will create a summarized report in ``results/report.md``.

