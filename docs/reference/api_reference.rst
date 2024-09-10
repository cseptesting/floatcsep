API Reference
=============

This contains a reference document to the floatCSEP API.


Commands
--------

The commands and entry-points with which to call `floatcsep` from the terminal
are:

.. :currentmodule:: floatcsep.commands.main

.. automodule:: floatcsep.commands.main

.. autosummary::
   :toctree: generated

    run
    plot
    reproduce


Experiment
----------

.. :currentmodule:: floatcsep.experiment

.. automodule:: floatcsep.experiment


The experiment is defined using the :class:`Experiment` class.

.. autosummary::
   :toctree: generated

    Experiment
    Experiment.set_models
    Experiment.get_model
    Experiment.stage_models
    Experiment.set_tests
    Experiment.catalog
    Experiment.set_test_cat
    Experiment.set_tasks
    Experiment.run
    Experiment.read_results
    Experiment.plot_results
    Experiment.plot_catalog
    Experiment.plot_forecasts
    Experiment.generate_report
    Experiment.to_dict
    Experiment.to_yml
    Experiment.from_yml


Models
------

.. :currentmodule:: floatcsep.model

.. automodule:: floatcsep.model

A model is defined using the :class:`Model` class.

.. autosummary::
   :toctree: generated

    Model
    Model.get_source
    Model.stage
    Model.init_db
    Model.rm_db
    Model.get_forecast
    Model.create_forecast
    Model.forecast_from_func
    Model.forecast_from_file
    Model.to_dict
    Model.from_dict


Evaluations
-----------

.. :currentmodule:: floatcsep.evaluation

.. automodule:: floatcsep.evaluation

A test is defined using the :class:`Evaluation` class.

.. autosummary::
   :toctree: generated

    Evaluation
    Evaluation.type
    Evaluation.get_catalog
    Evaluation.prepare_args
    Evaluation.compute
    Evaluation.write_result
    Evaluation.to_dict
    Evaluation.from_dict


Accessors
---------

.. :currentmodule:: floatcsep.utils.accessors

.. automodule:: floatcsep.utils.accessors

.. autosummary::
   :toctree: generated

    from_zenodo
    from_git
    download_file
    check_hash


Helper Functions
----------------

.. :currentmodule:: floatcsep.utils.helpers

.. automodule:: floatcsep.utils.helpers

.. autosummary::
   :toctree: generated

    parse_csep_func
    parse_timedelta_string
    read_time_config
    read_region_config
    timewindows_ti
    timewindows_td
    timewindow2str
    plot_sequential_likelihood
    magnitude_vs_time
    sequential_likelihood
    sequential_information_gain
    vector_poisson_t_w_test


Readers
-------

.. :currentmodule:: floatcsep.utils.readers

.. automodule:: floatcsep.utils.readers

.. autosummary::
   :toctree: generated

    ForecastParsers.dat
    ForecastParsers.xml
    ForecastParsers.quadtree
    ForecastParsers.csv
    ForecastParsers.hdf5
    HDF5Serializer.grid2hdf5
    serialize


Environments
------------

.. :currentmodule:: floatcsep.infrastructure.environments

.. automodule:: floatcsep.infrastructure.environments

.. autosummary::
   :toctree: generated

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


Registries
----------

.. :currentmodule:: floatcsep.infrastructure.registries

.. automodule:: floatcsep.infrastructure.registries

.. autosummary::
   :toctree: generated

    FileRegistry
    FileRegistry.abs
    FileRegistry.absdir
    FileRegistry.rel
    FileRegistry.rel_dir
    FileRegistry.file_exists

    ForecastRegistry
    ForecastRegistry.get
    ForecastRegistry.get_forecast
    ForecastRegistry.dir
    ForecastRegistry.fmt
    ForecastRegistry.as_dict
    ForecastRegistry.forecast_exist
    ForecastRegistry.build_tree
    ForecastRegistry.log_tree

    ExperimentRegistry
    ExperimentRegistry.add_forecast_registry
    ExperimentRegistry.get_forecast_registry
    ExperimentRegistry.log_forecast_trees
    ExperimentRegistry.get
    ExperimentRegistry.get_result
    ExperimentRegistry.get_test_catalog
    ExperimentRegistry.get_figure
    ExperimentRegistry.result_exist
    ExperimentRegistry.as_dict
    ExperimentRegistry.build_tree
    ExperimentRegistry.log_results_tree


Repositories
------------

.. :currentmodule:: floatcsep.infrastructure.repositories

.. automodule:: floatcsep.infrastructure.repositories

.. autosummary::
   :toctree: generated

   ForecastRepository
   ForecastRepository.factory

   CatalogForecastRepository
   CatalogForecastRepository.load_forecast
   CatalogForecastRepository._load_single_forecast

   GriddedForecastRepository.load_forecast
   GriddedForecastRepository._get_or_load_forecast
   GriddedForecastRepository._load_single_forecast

   ResultsRepository
   ResultsRepository._load_result
   ResultsRepository.load_results
   ResultsRepository.write_result

   CatalogRepository
   CatalogRepository.set_main_catalog
   CatalogRepository.catalog
   CatalogRepository.get_test_cat
   CatalogRepository.set_test_cat
   CatalogRepository.set_input_cat


Engine
------

.. :currentmodule:: floatcsep.infrastructure.engine

.. automodule:: floatcsep.infrastructure.engine

.. autosummary::
   :toctree: generated

    Task
    Task.sign_match
    Task.run
    Task.check_exist

    TaskGraph
    TaskGraph.ntasks
    TaskGraph.add
    TaskGraph.add_dependency
    TaskGraph.run
    TaskGraph.check_exist