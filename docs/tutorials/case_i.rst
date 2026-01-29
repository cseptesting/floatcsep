.. _case_i:

I — Containerizing a Model with Docker
======================================

**Goal.** Show how to containerize a forecasting model with **Docker** and run it inside
the floatCSEP engine, producing catalog forecasts and evaluating them with an N-test.
This case uses simple mock models as examples.

.. warning::

   **Docker is required** to containerize models and (optionally) to run in parallel.
   Please install Docker and complete the Linux post-installation steps if applicable.
   See :ref:`docker-install` in the Installation guide.

.. admonition:: **TL; DR**

    In a terminal, navigate to ``floatcsep/tutorials/case_i`` and type:

    .. code-block:: console

        $ floatcsep run config.yml

    After the calculation is complete, the results will be summarized in ``results/report.md``.
    The experiment region, catalog, forecasts and results can be viewed in the **Experiment Dashboard** with:

    .. code-block:: console

            $ floatcsep view config.yml


    For troubleshooting, run the experiment with:

    .. code-block:: console

        $ floatcsep run config.yml --debug

.. currentmodule:: floatcsep

.. contents:: Contents
    :local:



Experiment layout
-----------------

The experiment input files are:

::

    case_i
        └── pymock
            └── pymock  # src code
                | ...
            ├── Dockerfile  # Docker image build instructions
            └── setup.cfg  # Python build configuration
        └── pymock_slow  # Same as pymock, but slower (to test cpu and DAG usage)
            | ...
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
     start_date: 2012-5-23T00:00:00
     end_date:   2012-8-23T00:00:00
     horizon:    7days
     exp_class:  td

   region_config:
     region:    italy_csep_region
     mag_min:   3.5
     mag_max:   8.0
     mag_bin:   0.5
     depth_min: 0
     depth_max: 70

   run_mode:     parallel
   force_rerun:  True
   catalog:      catalog.csv
   model_config: models.yml
   test_config:  tests.yml

   postprocess:
     plot_forecasts: false

**Notes**

- ``exp_class: td`` declares a time-dependent experiment with rolling windows of length ``horizon``.
- ``run_mode: parallel`` enables parallel solving (if resources allow).
- ``force_rerun: True`` forces regeneration of forecasts even if files exist.


``models.yml``
^^^^^^^^^^^^^^

.. code-block:: yaml

   - pymock:
       path: pymock
       func: pymock
       func_kwargs:
         n_sims: 1000
         mag_min: 3.5
       build: docker

   - pymock_slow:
       path: pymock_slow
       func: pymock
       prefix: pymock
       func_kwargs:
         n_sims: 1000
         mag_min: 3.5
       build: docker

**Notes**

- ``build: docker`` tells floatCSEP to **build a Docker image** for the model located at ``path``.
- ``func`` is the entry-point used **inside** the container to run the forecasts (defined in ``setup.cfg``)
- ``func_kwargs`` are passed to ``model/input/args`` **inside** the container. They are stored and can be visualized in ``results/{time_window/input/{model}``.
- You can add the options ``force_build`` to rebuild the Docker image.


How containerization works here
-------------------------------

- For each model block marked ``build: docker``, floatCSEP:

  1. Builds a Docker image from the model directory at ``path`` (expects a valid Dockerfile and the model’s code/config there).
  2. Runs the container, mounting standard I/O folders used by the model (e.g., an ``input/`` and a model ``forecasts/`` directory), and executes your model’s entry-point (``func``).
  3. Collects the produced forecast files and proceeds with evaluations.


**Checklist to Dockerize your own model**

- Add a **Dockerfile** in your model folder (``path``).
- Ensure your entry-point can be invoked by the engine (e.g., a Python module or script that maps to ``func`` and accepts ``func_kwargs``).
- The model should be able to read input data (catalog and args) from ``{model_basedir}/input``
- Write outputs to the expected location (e.g., a ``{model_basedir}/forecasts/`` directory).
- (Optional): Avoid writing to root-owned paths inside the container; keep everything under the mounted work basedir.
- Test your image locally with a dry run:


from the ``{model_basedir}``

  .. code-block:: console

     $ docker build \
    --build-arg USERNAME=$USER \
    --build-arg USER_UID=$(id -u) \
    --build-arg USER_GID=$(id -g) \
    -t {model_name} .

     $ docker run --rm --volume $PWD:/usr/src/pymock:rw model_pymock python run.py input/args.txt


What happens under the hood
---------------------------

1. Controller parses rolling time windows from ``start_date`` to ``end_date`` with a 7-day horizon.
2. For each Dockerized model (``pymock``, ``pymock_slow``), the controller builds an image and launches
   containers with the appropriate inputs/arguments.
3. Forecasts are written to each model’s ``forecasts/`` directory (or central output).
4. The **Catalog N-test** runs on the collected forecasts and generates diagnostic plots.


Outputs
-------

You should find:

- Weekly gridded forecasts in each model’s ``forecasts/`` folder.
- N-test results (tables/JSON) in ``results/{time_window}/evaluations``.
- Markdown and PDF reports summarizing the experiment results in ``results/report.md``.


Running the experiment
----------------------

From the ``tutorials/case_i`` folder, run:

.. code-block:: console

   $ floatcsep run config.yml

This will build the Docker images (if needed), run the containers for each time window and model,
perform the evaluations, and create a summarized report in ``results/report.md``.


Troubleshooting
---------------

- Run in debug mode with ``floatcsep run config.yml --debug``, or add ``--log`` to output a log file ``results/log``
- **Docker not found / permission denied**:
  - Ensure Docker is installed and your user can run containers (Linux: add to ``docker`` group).
  - See :ref:`docker-install`.
- **Model image fails to build**:
  - Check a clean installation (without Docker) and running using its entry-point.
  - Check the Dockerfile, base image, and that all runtime deps are installed in the image.
  - Try building manually with ``docker build`` for clearer error messages.
  - Adapt one of the provided Dockerfiles in the model folders.
- **No forecasts produced**:
  - Confirm your model writes to the expected ``forecasts/`` directory.
  - Verify that the code actually reads from ``input/args.txt`` (or ``.yml`` or ``.json``).
- **Plots not generated**:
  - Inspect logs under ``results/`` for tracebacks.


pyCSEP under the hood
---------------------


    **Classes and functions used in this tutorial**

    - Catalog: :py:class:`csep.core.catalogs.CSEPCatalog`

        - :meth:`csep.load_catalog`
        - :meth:`csep.core.catalogs.CSEPCatalog.write_json`

    - Region: :py:class:`csep.core.regions.italy_csep_region`
    - Forecast class: :py:class:`csep.core.forecasts.CatalogForecast`

        - :meth:`csep.load_catalog_forecast`
        - :meth:`floatcsep.utils.file_io.CatalogForecastParsers.csv`

    - Test functions:

        - :py:func:`csep.core.catalog_evaluations.number_test`

    - Result plotting functions:

        - :py:func:`csep.utils.plots.plot_poisson_consistency_test`


    **Where to learn pyCSEP further:**

    - :doc:`pycsep:concepts/catalogs`
    - :doc:`pycsep:concepts/regions`
    - :doc:`pycsep:concepts/forecasts`
    - :doc:`pycsep:concepts/evaluations`