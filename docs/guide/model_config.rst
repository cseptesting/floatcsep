.. _model_config:


Models Configuration
====================

**floatCSEP** can integrate **source-code** models or just **forecast files**. Depending on the model type, configuration can be as simple as specifying a file path or as complex as defining the computational environment, run commands and model arguments. In the case of source-codes, the **Model Integration** section covers the environment management, executing the model code, and input/output dataflow.

In the experiment ``config.yml`` file (See :ref:`experiment_config`), the parameter ``model_config`` can point to a **model configuration** file, also in ``YAML`` format, with the generic structure:

**Example**:

   .. code-block:: yaml
      :caption: model_config.yml

      - MODEL_1 NAME:
          parameter_1: value
          parameter_2: value
          ...
      - MODEL_2 NAME:
          parameter_1: value
          parameter_2: value
          ...
      ...

Model names are used to identify models in the system, and spaces are replaced by underscores `_`.


Time-Independent Models
-----------------------

A **Time-Independent** model is usually represented by a single-file forecast, whose statistical description does not change over time.
Thus, the model configuration needs only to point to the **path** of the file relative to the ``model_config`` file.

**Example**:

.. code-block:: yaml

  - GEAR:
      path: models/gear.xml
      forecast_unit: 1

``forecast_unit`` represents the time frame upon which the forecast rates are defined (Defaults to 1). In time-independent forecasts, ``forecast_unit`` is in decimal **years**. Forecasts are scaled to the testing time-window if its length is different to the one of the forecast.



Time-Dependent Models
---------------------

**Time-Dependent** models are composed by forecasts issued for multiple time windows. These models can be either a **collection** of forecast files or a **source-code** that generate such collection.


1. **Forecast Collection**:

   In this case, the ``path`` must point to a model **directory**. To standardize with the directory structure of **source-code** models, forecasts should be contained in a folder named **forecasts** inside the model's ``path``.

   **Example**:

   .. code-block:: yaml

      - ETAS:
          path: models/etas
          forecast_unit: 3
          n_sims: 10000

   * Forecasts must be contained in a folder ``models/etas/forecasts``, relative to the ``model_config`` file.
   * The ``forecast_unit`` is defined in **days** for Time-Dependent models.
   * ``n_sims`` represents the total number of simulations from a catalog-based forecast (usually simulations with no events are not written, so the total amount of catalogs must be explicit).

   .. important::

      Forecast files are automatically detected. The standard way the model source should name a forecast is :

      .. code-block::

        {model_name}_{start}_{end}.csv

      where ``start`` and ``end`` follow either the ``%Y-%m-%dT%H:%M:%S`` - ISO8601 format, or the short date version ``%Y-%m-%d`` if the windows are set by UTC midnight.

   See the **pyCSEP** `Documentation <https://docs.cseptesting.org/concepts/forecasts.html#catalog-based-forecasts>`_ to see how forecast files should be written. See the :ref:`model_integration` section for details about how a model source-code should be designed or adapted to be integrated with **floatCSEP**

1. **Source-Code**:

   **floatCSEP** interacts with a model's source code by (i) creating a running environment, (ii) placing the input data (e.g., training catalog) within the model's directory structure, (iii) executing an specified run command and (iv) retrieving forecasts from the model directory structure. These actions will be detailed in the :ref:`model_integration` section.

   The basic parameters of the configuration are:

   *  ``path`` refers to the source-code directory.
   * The ``build`` parameter defines the environment type (e.g., ``conda``, ``venv``, or ``docker``) and ensures the model runs in isolation with the necessary dependencies.
   * ``func`` is a `shell` command (**entrypoint**) with which the source-code is executed inside the environment.
   * The ``forecast_unit`` is defined in **days** for Time-Dependent models.

   **Example**:

   .. code-block:: yaml

      - STEP:
          path: models/step
          build: docker
          func: etas-run
          forecast_unit: 1

Repository Download
-------------------

A model file(s) or source code can be accessed from a code or data repository (i.e., `GitHub <https://github.com>`_ or `Zenodo <https://zenodo.org>`_).

.. code-block:: yaml

   - etas:
       giturl: https://git.gfz-potsdam.de/csep/it_experiment/models/vetas.git
       repo_hash: v3.2

where ``repo_hash`` refers to a given **release**, **tag** or **branch**. Alternatively, a model can be retrieved from a Zenodo repository by specifying its ID:

.. code-block:: yaml

   - wheel:
       zenodo_id: 6255575



Configuration Parameters
------------------------

Here you can find a comprehensive list of parameters used to configure models

.. list-table::
   :widths: 20 20 60
   :header-rows: 1

   * - **Name**
     - **Type**
     - **Description**
   * - **path** (required)
     - All
     - Path to the model’s (i) **forecast file** for a time-independent class, or (ii) **model's directory** for time-dependent class
   * - **build**
     - TD
     - Specifies the environment type in which the model will be built (e.g., ``conda``, ``venv``, ``docker``).
   * - **zenodo_id**
     - All
     - Zenodo record ID for downloading the model's data.
   * - **giturl**
     - All
     - Git repository URL for the model’s source code.
   * - **repo_hash**
     - All
     - Specifies the commit, branch, or tag to be checked out from the repository.
   * - **args_file** (required)
     - TD
     - Path to the input arguments file for the model, relative to ``path``. In here, the forecast start_date and end_date will be dynamically written before each forecast creation. Defaults to ``input/args.txt``.
   * - **func**
     - TD
     - The command to execute the model (i.e., **entrypoint**) in a terminal. Examples of ``func`` are: ``run``, ``etas-run``, ``python run_script.py``, ``Rscript script.r``.
   * - **func_kwargs** (optional)
     - TD
     - Additional arguments for the model execution, passed via the arguments file.
   * - **forecast_unit** (required)
     - All
     - Specifies the time unit for the forecast. Use **years** for time-independent models and **days** for time-dependent models.
   * - **store_db** (optional)
     - All
     - If the model consists on only files, this is a boolean (true/false) specifying whether to store the forecast in a database (HDF5).
   * - **flavours** (optional)
     - All
     - A set of parameter variations to generate multiple model variants (e.g., different settings for the same model).
   * - **prefix** (optional)
     - TD
     - The prefix used for the model to name its forecast (The default is the Model's name)
   * - **input_cat** (optional)
     - TD
     - Specifies the input catalog path used by the model, relative to the model's ``path``. Defaults to ``input/catalog.csv``.
   * - **force_stage** (optional)
     - All
     - Forces the entire staging of the model (e.g., downloading data, database preparation, environment creation, installation of dependencies and source-code build)
   * - **force_build** (optional)
     - All
     - Forces the build of the model's environment (e.g., creation, dependencies installation  and source-code build)



.. _model_integration:

Model Integration
-----------------

The integration of external model source-codes into **floatCSEP** requires:

* Follow (loosely) a directory structure to allow the dataflow (input/output) between the model and **pyCSEP**.
* Define a environment/container manager.
* Provide source-code build instructions.
* Set up an entrypoint (terminal command) to run the model and create a forecast.

.. note::

    To integrate a broader range of model classes and code complexities, we opted in **floatCSEP** for a simple interface design rather than specifying a complex model API. Therefore, the integration will have sometimes strict requirements, or customizable options and sometimes undefined aspects. We encourage any feedback from modelers (and hopefully their contributions) through our GitHub, to encompass the majority of model implementations possible.

Directory Structure
~~~~~~~~~~~~~~~~~~~

The repository should contain, at the least, the following structure:

.. code-block:: none

    model_name/
    ├── /forecasts          # Forecast outputs should be stored here (Required)
    ├── /input              # Input data will be placed here dynamically by **floatCSEP** (Required)
    │   ├── {input_catalog} # Input catalog file provided by the testing center
    │   └── {args_file}     # Contains the input arguments for model execution
    ├── /{source}           # [optional] Where to store all the source code of the model
    │   └── ...
    ├── /state              # [optional] State files (e.g., data to be persisted throughout consistent simulations)
    ├── README.md           # [optional] Basic information of the model and instructions to run it.
    ├── {run_script}        # [optional] Script to generate forecasts. Can be either located here, or in the environment PATH (e.g., a binary entrypoint for python)
    ├── Dockerfile          # Docker environment setup file
    ├── environment.yml     # Instructions to build a conda environment.
    └── setup.py            # Script to build the code with "pip install . ". Can also be `project.toml` or `setup.cfg`


* The name of the files ``input_catalog`` (default: `catalog.csv`) and ``args_file`` (default: `args.txt`) can be controlled within ``model_config``.
* It is required (for this integration protocol) that the folders ``input`` and ``forecasts`` exists in the model directory. The latter could be created during the first model run.

.. important::
    The directory structure should remain unchanged during the experiment run, except for the dynamic modification of the `input/`, `forecasts/` and `state/` contents. All of the source-code file management routines should point to these folders (e.g., routines to read input catalogs, read input arguments, to write forecasts, etc.).


Environment Management
~~~~~~~~~~~~~~~~~~~~~~

The `build` parameter in the model configuration specifies the environment type (e.g., `conda`, `venv`, `docker`). Models should be defined in an isolated environment to ensure reproducibility and prevent conflicts with system dependencies.

1. **venv**: A Python virtual environment (`venv`) setup is specified. The source code will be built by running the command ``pip install .`` within the virtual sub-environment (an environment within the one **floatCSEP** is run, but isolated from it), pointing to a ``setup.py``, ``setup.cfg`` or ``project.toml`` (See the `Packaging guide <https://packaging.python.org/en/latest/guides/writing-pyproject-toml>`_)

2. **conda**: The model sub-environment is managed via a `conda` environment file (``environment.yml``). The model source-code will still be built using ``pip``.

3. **docker**: A Docker container is created based on a provided `Dockerfile` that contains the instruction to build the source-code within.(`Writing a Dockerfile <https://docs.docker.com/get-started/docker-concepts/building-images/writing-a-dockerfile/>`_). If python, the model source-code will still be built using ``pip`` inside a virtual environment.

.. note::
    All the environment names will be handled internally by **floatCSEP**.

**Example setup.cfg**


.. code-block:: cfg

    [metadata]
    name = cookie_model
    description = Just another model
    author = Monster, Cookie

    [options]
    packages =
        cookie_model
    install_requires =
        numpy
    python_requires = >=3.9

    [options.entry_points]
    console_scripts =
        cookie-run = cookie_model.main:run

This build configuration installs the dependencies (``numpy``), the module ``cookie_model`` (i.e., the ``{source}`` folder) and creates an entrypoint command (see the :ref:`model_execution` section).



**Example Dockerfile**

.. code-block:: dockerfile

    # Use a specific Python version from a trusted source
    FROM python:3.9.20

    # Set up user and permissions
    ARG USERNAME=modeler
    ARG USER_UID=1100
    RUN useradd -u $USER_UID -m -s /bin/sh $USERNAME

    # Set work directory
    WORKDIR /usr/src/

    # Copy repository contents to the container
    COPY --chown=$USERNAME cookie_model ./cookie_model/
    COPY --chown=$USERNAME setup.cfg ./

    # Install the Python package and upgrade pip
    RUN pip install --no-cache-dir --upgrade pip && pip install .

    # Set the default user
    USER $USERNAME


This Dockerfile will install the python package inside a container, but the concept can be applied also for other programming languages. The ``func`` parameter will be used identically as done for ``conda`` and ``venv`` options, but now **floatCSEP** will handle the container execution and the entrypoint.


.. _model_execution:

Model Entrypoint
~~~~~~~~~~~~~~~~

A model should be executed always with a shell command through a terminal. This provides flexibility to the modeler to abstract their model as convenient.
The **func** parameter in the model configuration defines the shell command used to execute the model. This command is invoked within the environment set up by **floatCSEP**, and will be run from ``model_path`` or the entrypoint defined in the ``Dockerfile``.

Example ``func`` commands:

.. code-block:: console

    $ cookie-run
    $ python run.py
    $ Rscript run.R
    $ sh run.sh

The ``cookie-run`` was a binary python entrypoint defined in the previous **Example setup.cfg**. It allows to execute the command ``cookie-run`` from the terminal, which itself will run the `python` function :func:`cookie_model.main.run` from the file ``cookie_model/main.py``.

.. note::

    This entrypoint function should contain the high-level logic of the model workflow (e.g, reading input, parsing arguments, calling core routines, write forecasts, etc.). An example pseudo-code of a model's workflow is:

    .. code-block:: R

       start, end, args = read_input(args_path)
       training_catalog = read_catalog(input_cat)
       parameters = fit(training_catalog)
       forecast = create_forecast(start, end, args, parameters)
       write(forecast)



Input/Output Dataflow
~~~~~~~~~~~~~~~~~~~~~

The input to run a model will be placed into the ``model_path/input/`` directory dynamically by the testing system before each model execution. The model should be able to read these files from this directory. Similarly, after each model execution, the resulting forecast should be stored in a ``model_path/forecasts/`` directory

We distinguish **input data** versus **input arguments**. The input data is given to a model without control of the modeler (e.g. authoritative input catalog, region), whereas input arguments (as in *function* arguments) can be the forecast specifications (e.g. time-window, target magnitudes) or hyper-parameters (e.g. declustering algorithm, optimization time-windows, cutoff magnitude) that control the model.


1. **Input Arguments**: The input arguments are the forecast specifications (e.g. time-window, target magnitudes) and hyper-parameters (e.g. declustering algorithm, optimization time-windows, cutoff magnitude) that will control the model. The input arguments will be written in the ``args_file`` (default ``args.txt``) always located in the input folder. A model requires at minimum one set of modifiable arguments: ``start_date`` and ``end_date`` (in ISO8601), but it is possible to include additional arguments.

   Example content of ``args.txt``:

   .. code-block:: yaml

      start_date: 2023-01-01T00:00:00
      end_date: 2023-01-02T00:00:00
      seed: 23
      nsims: 1000

   Therefore, the model source-code should be at least able to dynamically read the obligatory arguments (simply the time window of the issued forecast)

2. **Input Data**: Correspond to any data source outside the control of the modeler (e.g., authoritative input catalog, testing region). For now, **floatCSEP** just handles an input **catalog**, which are all the events within the **main catalog**  until the forecast **start_date**. The catalog is written by default in ``model_path/input/catalog.csv`` in the CSEP ascii format (see :doc:`pycsep:concepts/catalogs`) as:

  .. code-block:: none

      longitude, latitude, magnitude, time_string, depth, event_id

  - **longitude**: Decimal degrees of the forecasted event location.
  - **latitude**: Decimal degrees of the forecasted event location.
  - **magnitude**: Magnitude of the forecasted event.
  - **time_string**: Timestamp in UTC following the ISO8601 format (`%Y-%m-%dT%H:%M:%S`).
  - **depth**: Depth of the event in kilometers.
  - **event_id**: The event ID in case is necessary to map the event to an additional table.


3. **Output Forecasts**: After execution, forecast files should be written to the `forecasts/` folder. The forecast output must follow the filename convention:

   .. code-block:: none

      {model_name}_{start-date}_{end-date}.csv

  ``model_name`` can be replaced in the model configuration with the parameter ``prefix``, such that:

  .. code-block:: none

      {prefix}_{start-date}_{end-date}.csv


  This ensures that forecast files are easily identified and retrieved by **floatCSEP** for further evaluation.


  .. important::

     The forecast files should adhere to the **pyCSEP** format. In summary, each forecast file should be a ``.csv`` file containing rows for each forecasted event, whose columns are:

     .. code-block:: none

        longitude, latitude, magnitude, time_string, depth, catalog_id, event_id

     where catalog_id represents the a single simulation of the stochastic catalog collection. This format ensures compatibility with the **pyCSEP** testing framework (See the `Catalog-based forecasts <https://docs.cseptesting.org/concepts/forecasts.html#working-with-catalog-based-forecasts>`_ documentation for further information).





