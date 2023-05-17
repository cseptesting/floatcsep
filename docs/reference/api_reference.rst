API Reference
=============

This contains a reference document to the floatCSEP API.


Commands
--------

The commands and entry-points with which to call `floatcsep` from the terminal
are:

.. :currentmodule:: floatcsep.cmd.main

.. automodule:: floatcsep.cmd.main

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

.. :currentmodule:: floatcsep.accessors

.. automodule:: floatcsep.accessors

.. autosummary::
   :toctree: generated

    query_gcmt
    from_zenodo
    from_git



Extras
------

Additional `pyCSEP` functionalities


.. :currentmodule:: floatcsep.extras

.. automodule:: floatcsep.extras

.. autosummary::
   :toctree: generated

    sequential_likelihood
    sequential_information_gain
    vector_poisson_t_w_test
    brier_score
    negative_binomial_number_test
    binomial_joint_log_likelihood_ndarray
    binomial_spatial_test
    binomial_conditional_likelihood_test
    binary_paired_t_test
    log_likelihood_point_process
    paired_ttest_point_process



Utilities
---------

.. :currentmodule:: floatcsep.utils

.. automodule:: floatcsep.utils

.. autosummary::
   :toctree: generated

    parse_csep_func
    parse_timedelta_string
    read_time_config
    read_region_config
    timewindows_ti
    timewindows_td
    Task
    Task.run
    Task.check_exist
    timewindow2str
    plot_sequential_likelihood
    magnitude_vs_time



Readers
-------

.. :currentmodule:: floatcsep.readers

.. automodule:: floatcsep.readers

.. autosummary::
   :toctree: generated

    ForecastParsers.dat
    ForecastParsers.xml
    ForecastParsers.quadtree
    ForecastParsers.csv
    ForecastParsers.hdf5
    HDF5Serializer.grid2hdf5
    serialize


