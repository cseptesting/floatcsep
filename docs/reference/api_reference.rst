API Documentation
=================


.. Here are the listed and linked the rst pages of the API docs. Hidden means it wont show on
.. this api reference landing page.

.. toctree::
   :maxdepth: 1
   :hidden:

   commands
   experiment
   model
   evaluation
   postprocess
   utilities
   infrastructure


.. Here we create fake autosummaries, which are excluded in the conf.py, so they are not shown
.. in the documentation, but still we are displaying the neat summary tables for each classs

**Commands**

.. currentmodule:: floatcsep.commands.main

The main entrypoint functions from the Command Line Interface are:

.. autosummary::
   :nosignatures:

    floatcsep
    run
    stage
    plot
    reproduce


**Experiment**

.. currentmodule:: floatcsep.experiment

The :class:`~floatcsep.experiment.Experiment` class is the main handler of floatCSEP, which
orchestrates the :class:`~floatcsep.model.Model` and :class:`~floatcsep.evaluation.Evaluation`
instances onto an experimental workflow. The class and its main methods are:

.. autosummary::
   :nosignatures:

    Experiment
    Experiment.set_models
    Experiment.set_tests
    Experiment.stage_models
    Experiment.set_input_cat
    Experiment.set_test_cat
    Experiment.set_tasks
    Experiment.run
    Experiment.read_results
    Experiment.make_repr

**Model**

.. currentmodule:: floatcsep.model

The :class:`~floatcsep.model.Model` class is the handler of forecasts creation, storage and
reading. The abstract and concrete classes, and their main methods are:

.. autosummary::
   :nosignatures:

   Model
   Model.get_source
   Model.factory

   TimeIndependentModel
   TimeIndependentModel.init_db
   TimeIndependentModel.get_forecast

   TimeDependentModel.stage
   TimeDependentModel.prepare_args
   TimeDependentModel.create_forecast
   TimeDependentModel.get_forecast


**Evaluations**

.. currentmodule:: floatcsep.evaluation

The :class:`~floatcsep.evaluation.Evaluation` class is a wrapper for `pycsep` functions,
encapsulating the multiple function, arguments, forecast and catalogs of the entire experiment.
The class and main methods are:

.. autosummary::
   :nosignatures:

    Evaluation
    Evaluation.prepare_args
    Evaluation.compute

**Accessors**

These are functions that access a model source from a web repository.

.. currentmodule:: floatcsep.utils.accessors

.. autosummary::
   :nosignatures:

    from_zenodo
    from_git
    download_file
    check_hash


**Helper Functions**

These are the helper functions of ``floatCSEP``

.. currentmodule:: floatcsep.utils.helpers

.. autosummary::
   :nosignatures:

    parse_csep_func
    timewindow2str
    str2timewindow
    parse_timedelta_string
    read_time_cfg
    read_region_cfg
    timewindows_ti
    timewindows_td


Some additional plotting functions to pyCSEP are:

.. autosummary::
   :nosignatures:

    plot_sequential_likelihood
    magnitude_vs_time
    sequential_likelihood
    sequential_information_gain
    vector_poisson_t_w_test


**Readers**

A small wrapper for ``pyCSEP`` readers

.. currentmodule:: floatcsep.utils.readers

.. autosummary::
   :nosignatures:

    ForecastParsers.dat
    ForecastParsers.xml
    ForecastParsers.quadtree
    ForecastParsers.csv
    ForecastParsers.hdf5
    HDF5Serializer.grid2hdf5
    serialize


**Environments**

The computational environment managers for ``floatcsep``.

.. currentmodule:: floatcsep.infrastructure.environments

.. autosummary::
   :nosignatures:

    CondaManager
    CondaManager.create_environment
    CondaManager.env_exists
    CondaManager.install_dependencies
    CondaManager.run_command

    VenvManager
    CondaManager.create_environment
    CondaManager.env_exists
    CondaManager.install_dependencies
    CondaManager.run_command


**Registries**

The registries hold references to the access points (e.g., filepaths) of the experiment
components (e.g., forecasts, catalogs, results, etc.), and allows to be aware of their status.

.. currentmodule:: floatcsep.infrastructure.registries

.. autosummary::
   :nosignatures:

    ForecastRegistry
    ForecastRegistry.get_forecast
    ForecastRegistry.fmt
    ForecastRegistry.forecast_exists
    ForecastRegistry.build_tree

    ExperimentRegistry
    ExperimentRegistry.add_forecast_registry
    ExperimentRegistry.get_forecast_registry
    ExperimentRegistry.get_result
    ExperimentRegistry.get_test_catalog
    ExperimentRegistry.get_figure
    ExperimentRegistry.result_exist
    ExperimentRegistry.build_tree


**Repositories**

The repositories here are designed to store and access the experiment artifacts (results,
catalogs, forecasts), abstracting the experiment logic from the pyCSEP io functionality.


.. currentmodule:: floatcsep.infrastructure.repositories

.. autosummary::
   :nosignatures:

   CatalogRepository
   CatalogRepository.set_main_catalog
   CatalogRepository.catalog
   CatalogRepository.get_test_cat
   CatalogRepository.set_test_cat
   CatalogRepository.set_input_cat

   GriddedForecastRepository
   GriddedForecastRepository.load_forecast

   CatalogForecastRepository
   CatalogForecastRepository.load_forecast

   ResultsRepository
   ResultsRepository.load_results
   ResultsRepository.write_result



**Engine**

The engine routines are designed for the execution of an experiment.

.. currentmodule:: floatcsep.infrastructure.engine

.. autosummary::
   :nosignatures:

    Task
    Task.run
    Task.sign_match

    TaskGraph
    TaskGraph.ntasks
    TaskGraph.add
    TaskGraph.add_dependency
    TaskGraph.run
