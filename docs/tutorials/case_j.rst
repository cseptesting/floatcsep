.. _case_j:

J — Running in Parallel a Real Model
====================================

**Goal.** Run a time-dependent experiment over New Zealand using the STEP model with a Docker build,
producing daily gridded forecasts and evaluating them with a Poisson N-test. The experiment is solved in parallel.

.. warning::

   **Docker is required** to containerize models and to run experiments in parallel.
   Please install Docker and complete the Linux post-installation steps if applicable.
   See :ref:`docker-install` in the Installation guide.

.. admonition:: **TL; DR**

    In a terminal, navigate to ``floatcsep/tutorials/case_j`` and type:

    .. code-block:: console

        $ floatcsep run config.yml

    After the calculation is complete, the results will be summarized in ``results/report.md``.
    The experiment region, catalog, forecasts and results can be viewed in the **Experiment Dashboard** with:

    .. code-block:: console

            $ floatcsep view config.yml
            $ floatcsep view config.yml --ui nextjs  # Alternative: modern Next.js dashboard


    For troubleshooting, run the experiment with:

    .. code-block:: console

        $ floatcsep run config.yml --debug


.. currentmodule:: floatcsep

.. contents:: Contents
    :local:



Directory layout
----------------

The experiment input files are:

::

    case_h
        ├── catalog.csep
        ├── config.yml
        ├── tests.yml
        └── models.yml


Configuration
-------------

``config.yml``
^^^^^^^^^^^^^^

.. code-block:: yaml

   name: case_i

   time_config:
     start_date: 2016-11-13T00:00:00
     end_date:   2016-11-20T00:00:00
     horizon:    1days
     exp_class:  td

   region_config:
     region:    nz_csep_region
     mag_min:   3.0
     mag_max:   8.0
     mag_bin:   0.1
     depth_min: 0
     depth_max: 70

   run_mode:     parallel
   force_rerun:  True
   catalog:      catalog.csep
   model_config: models.yml
   test_config:  tests.yml

**Notes**
- ``exp_class: td`` declares a time-dependent (operational) experiment with rolling 1-day windows.
- ``run_mode: parallel`` runs the experiment in parallel.
- ``force_rerun: True`` will force the models to recreate the forecasts even if the exist in the fileysystem.

``models.yml``
^^^^^^^^^^^^^^

.. code-block:: yaml

   - step:
       giturl: git@github.com:KennyGraham1/STEPModel.git
       repo_hash: floatcsep-interface
       forecast_type: gridded
       fmt: dat
       path: models/step
       args_file: step_config.yaml
       build: docker
       func: run
       func_kwargs:
         paths:
           catalog_file: catalog.csv
         forecast:
           min_magnitude: 3.0
           region_code: 90
           parameters:
             a_value: -1.00
             b_value: 1.03
             p_value: 1.07
             c_value: 0.04

**Notes**
- ``build: docker`` — the engine will build a Docker image from the repo.
- ``fmt: dat`` specifies the created forecast, which are interpreted as GriddedForecasts


What happens under the hood
---------------------------

1. Controller parses time windows from ``2016-11-13`` to ``2016-11-20`` (1-day horizon).
2. Clones the STEP repo with a given ``repo_hash``  into ``models/step/`` and builds a Docker image.
3. For each 1-day window, mounts the required folders, provides the input catalog and arguments, and runs ``func: run``.
4. Collects daily **gridded** forecasts (``.dat``) and runs the Poisson N-test with the configured plot.


Outputs
-------

You should find:

- Daily gridded forecasts (``.dat``) in the model’s ``forecasts/`` folder
- Forecast plots in ``results/{time_window}/figures``
- N-test results (table/JSON) in ``results/{time_window}/evaluations``
- A Markdown and PDF reports summarizing the experiment resul.


Running the experiment
----------------------

    The experiment can be run by simply navigating to the ``tutorials/case_h`` folder in the terminal and typing:

    .. code-block:: console

       $ floatcsep run config.yml

    This will automatically set all the calculation paths (testing catalogs, evaluation results, figures) and will create a summarized report in ``results/report.md``.


pyCSEP under the hood
---------------------


    **Classes and functions used in this tutorial**

    - Catalog: :py:class:`csep.core.catalogs.CSEPCatalog`

        - :meth:`csep.load_catalog`
        - :meth:`csep.core.catalogs.CSEPCatalog.write_json`

    - Region: :py:class:`csep.core.regions.nz_csep_region`
    - Forecast class: :py:class:`csep.core.forecasts.GriddedForecast`

        - :meth:`csep.load_gridded_forecast`
        - :meth:`floatcsep.utils.file_io.GriddedForecastParsers.dat`

    - Test functions:

        - :py:func:`csep.core.poisson_evaluations.number_test`

    - Result plotting functions:

        - :py:func:`csep.utils.plots.plot_poisson_consistency_test`


    **Where to learn pyCSEP further:**

    - :doc:`pycsep:concepts/catalogs`
    - :doc:`pycsep:concepts/regions`
    - :doc:`pycsep:concepts/forecasts`
    - :doc:`pycsep:concepts/evaluations`