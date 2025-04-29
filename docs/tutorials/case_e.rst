.. _case_e:

E - A Time-Independent Experiment
=================================

This example shows how to run a realistic testing experiment (based on :ref:`Schorlemmer et al. 2010<References>`) while summarizing the concepts from the previous tutorials.

.. currentmodule:: floatcsep

.. admonition:: **TL; DR**

    In a terminal, navigate to ``floatcsep/tutorials/case_e`` and type:

    .. code-block:: console

        $ floatcsep run config.yml

    After the calculation is complete, the results will be summarized in ``results/report.md``.

.. contents:: Contents
    :local:



Experiment Components
---------------------

The source code can be found in the ``tutorials/case_e`` folder or in `the GitHub repository <https://github.com/cseptesting/floatcsep/blob/main/tutorials/case_e>`_. The input structure of the experiment is:

::

    case_e
        └──  models
            ├── gulia-wiemer.ALM.italy.10yr.2010-01-01.xml
            ├── meletti.MPS04.italy.10yr.2010-01-01.xml
            └── zechar.TripleS-CPTI.italy.10yr.2010-01-01.xml
        ├── config.yml
        ├── models.yml
        └── tests.yml

.. note::
    This experiment has only a subset of the original models and evaluations.


Configuration
-------------


Time
~~~~

    The time configuration is manifested in the ``time-config`` inset.

    .. literalinclude:: ../../tutorials/case_e/config.yml
        :caption: tutorials/case_e/config.yml
        :language: yaml
        :lines: 3-7

Region
~~~~~~

    The testing region is the official Italy CSEP Region obtained from :obj:`csep.core.regions.italy_csep_region`.

    .. literalinclude:: ../../tutorials/case_e/config.yml
        :caption: tutorials/case_e/config.yml
        :language: yaml
        :lines: 9-15


Catalog
~~~~~~~

    The catalog is obtained from an authoritative source, namely the Bollettino Sismico Italiano (http://terremoti.ingv.it/en/bsi ), using the function :func:`csep.query_bsi`

    .. literalinclude:: ../../tutorials/case_e/config.yml
        :caption: tutorials/case_e/config.yml
        :language: yaml
        :lines: 17-17

Models
~~~~~~
    The models are set in ``models.yml``. For simplicity, there are only three of the nineteen models originally submitted to the Italy Experiment.

    .. literalinclude:: ../../tutorials/case_e/models.yml
        :caption: tutorials/case_e/models.yml
        :language: yaml

    The ``.xml`` format is automatically detected and parsed by ``floatcsep`` readers.

    .. important::

        The forecasts are defined in ``[Earthquakes / 10-years]``, which is specified with the ``forecast_unit`` option (The default is `forecast_unit: 1`).

    .. note::

        The ``use_db`` flag allows ``floatcsep`` to transform the forecasts into a database (HDF5), which speeds up the calculations.

Post-Process
~~~~~~~~~~~~

    Additional options for post-processing can set using the ``postprocess`` option. Here, we customize the forecasts plotting with:

    .. literalinclude:: ../../tutorials/case_e/config.yml
       :language: yaml
       :lines: 21-34

    The forecasts are plotted and stored in ``tutorials/case_e/results/{timewindow}/forecasts/``. See :func:`~csep.utils.plots.plot_spatial_dataset` for forecast plotting options and :func:`~csep.utils.plots.plot_catalog` for the catalog placed on top of those plots.


Running the experiment
----------------------

    The experiment can be run by navigating to the ``tutorials/case_e`` folder in the terminal and typing.

    .. code-block:: console

        $ floatcsep run config.yml

    This will automatically set all the calculation paths (testing catalogs, evaluation results, figures) and will create a summarized report in ``results/report.md``.


Plot command
~~~~~~~~~~~~

    If only the result plots are desired when the calculation was already completed before, you can type:

    .. code-block:: console

        $ floatcsep plot config.yml

    This can be used, for example, when an additional plot is desired without re-running the entire experiment. Try adding the following line to the ``postprocess`` inset of the ``config.yml`` file.

    .. code-block:: yaml

        postprocess:
          plot_forecasts:
            colormap: magma

    and re-run with the ``plot`` command. A forecast figure will re-appear in ``results/{window}/forecasts`` with a different colormap. Additional forecast and catalog plotting options can be found in the :func:`csep.utils.plots.plot_spatial_dataset` and :func:`csep.utils.plots.plot_catalog` ``pycsep`` functions.


References
----------

    * Schorlemmer, D., Christophersen, A., Rovida, A., Mele, F., Stucchi, M. and Marzocchi, W. (2010). Setting up an earthquake forecast experiment in Italy. Annals of Geophysics, 53(3), 1–9. doi: `10.4401/ag-4844 <https://doi.org/10.4401/ag-4844>`_
