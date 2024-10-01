.. _running:

Executing an Experiment
=======================

In **floatCSEP**, experiments are executed through a set of core functions provided in the command-line interface (CLI). These commands allow you to stage models, run experiments, plot results, and generate reports. The general structure of these commands is:

.. code-block:: console

    $ floatcsep <command> <config.yml>


The `<command>` can be one of the following:
- `run`: Run the experiment.
- `stage`: Prepare the environment and models.
- `plot`: Plot forecasts, catalogs, and results.
- `reproduce`: (see folllowing\ section) Reproduce the results of a previously run experiment.

Each command requires a configuration file (in `YAML` format), which defines the parameters of the experiment.

Running an Experiment: ``floatcsep run``
----------------------------------------

The core command to run an experiment is:

.. code-block:: console

    $ floatcsep run <config.yml>


This command initiates the workflow summarized as:

1. **Experiment Initialization**: The experiment is initialized by reading the configuration file (`config.yml`). This file contains details such as time windows, regions, catalogs, models, and evaluation tests.

2. **Staging Models**: If the models are not already staged, they are fetched from their respective repositories (e.g., GitHub, Zenodo) or located on the file system. The necessary computational environments are built using tools like **Conda**, **venv**, or **Docker**.

3. **Task Generation**: Depending on the experiment class, a given acyclic graph of tasks is created (e.g., a collection of tasks with dependencies to one another). These tasks include creating forecasts, filtering catalogs, and evaluating the forecasts using statistical tests.

4. **Execution**: The task graph is executed on a standardized fashion. Depending on the characteristics of the experiment (e.g., time-dependent, evaluation windows), this step might involve generating forecasts and running evaluations in sequence.

5. **Postprocessing**: After the core tasks are completed, the postprocessing step involves:
   - Plotting forecasts, catalogs, and evaluation results.
   - Generating human-readable reports, which summarize the results.

6. **Reproducibility**: The configuration and results are saved, allowing the experiment to be reproduced or compared in future runs.

Here is an example of how to run an experiment:

.. code-block:: sh

    $ floatcsep run config.yml


For more information on how to structure the configuration file, refer to the :ref:`experiment_config` section.


Staging Models: ``floatcsep stage``
-----------------------------------

Before running an experiment, you may need to check if the models are `staged` properly instead. This involves fetching the source code for the models from a repository (e.g., GitHub, Zenodo) and setting up the necessary computational environment.

.. code-block:: console

    floatcsep stage <config.yml>


Staging the models can be done previous to an experiment run when dealing with source-code models that need specific environments and dependencies to run properly. The `staging` process includes:

- Downloading the source code.
- Building the computational environment (e.g., setting up a Conda environment or Docker container).
- Installing any dependencies required by the models.
- Building the source code.
- Check a correct integration with floatcsep.
- Prepare the structure of the required forecast.
- Self-discovery of existing forecasts in the filesystem.

 it does plot the results of an already completed experiment.

.. note::

    This command should be executed to check if everything is present and working correctly before an official ``run`` execution.

Plotting Results: ``floatcsep plot``
------------------------------------

Once the experiment has been run, you can regenerate plots for the forecasts, catalogs, and evaluation results using the `plot` command:

.. code-block:: console

    $ floatcsep plot <config.yml>


The `plot` command re-loads the experiment configuration, stages the models, identifying the necessary time windows and results to plot. It does not re-run the forecasts or evaluations, but it does plot the results of an already completed experiment.

.. note::

    This command can be useful when trying to customize plots or reports after the results have been created.


Reproducing Results: ``floatcsep reproduce``
--------------------------------------------

The `reproduce` command in **floatCSEP** allows users to re-run a previously executed experiment using the same configuration, in order to compare the results and assess reproducibility. This feature allows to ensure that experiments yield consistent outputs when re-executed and to validate scientific results.

The general command structure is:

.. code-block:: console

    $ floatcsep reproduce <repr_config.yml>

A ``repr_config.yml`` is always generated once an experiment is run with ``floatcsep run``. The ``reproduce`` command re-runs the experiment based on this configuration and compares the newly generated results with the original results to provide reproducibility metrics:

- **Statistical Reproducibility**: It analyzes statistical changes of the evaluation results:

  - **Forecast Scores**: The numerical difference between the observed scores of the original and reproduced experiments.
  - **Test Statistics**: Statistical metrics like mean, standard deviation, and skewness of the test distributions are compared.
  - **Kolmogorov-Smirnov (KS) Test**: The KS-test p-value is computed to assess whether the test distributions from both experiments are significantly different. A p-value below 0.1 indicates a potential difference between distributions.

- **Data Reproducibility**: A comparison of the result files, checking for discrepancies in file contents or structure.

  - **Hash Comparison (SHA-256)**: Each result file is hashed using the SHA-256 algorithm to check if the content has changed between the original and reproduced experiments.
  - **Byte-to-Byte Comparison**: This is a direct comparison of the file contents at the byte level, ensuring that no unintended changes have occurred.


The analysis can now be found in the created ``reproducibility_report.md``.
