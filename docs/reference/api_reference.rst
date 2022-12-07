API Reference
=============

This contains a reference document to the feCSEP API.


Experiment
----------

.. :currentmodule:: fecsep.experiment

.. automodule:: fecsep.experiment


The experiment is defined using the :class:`Experiment` class.

.. autosummary::
   :toctree: generated

    Experiment
    Experiment.set_models
    Experiment.set_tests
    Experiment.prepare_paths
    Experiment.prepare_tasks
    Experiment.run
    Experiment.plot_results
    Experiment.plot_forecasts
    Experiment.generate_report
    Experiment.from_yml
    Experiment.to_dict
    Experiment.to_yml


Model
-----

.. :currentmodule:: fecsep.model

.. automodule:: fecsep.model

A model is defined using the :class:`Model` class.

.. autosummary::
   :toctree: generated

    Model
    Model.get_source
    Model.init_db
    Model.rm_db
    Model.get_forecast
    Model.create_forecast
    Model.forecast_from_func
    Model.forecast_from_file
    Model.to_dict
    Model.from_dict


Test
----

.. :currentmodule:: fecsep.evaluation

.. automodule:: fecsep.evaluation

A test is defined using the :class:`Evaluation` class.

.. autosummary::
   :toctree: generated

    Evaluation
    Evaluation.compute
    Evaluation.type
    Evaluation.to_dict
    Evaluation.from_dict




Running an Experiment
----------------------

.. :currentmodule:: fecsep.cmd.main

.. automodule:: fecsep.cmd.main

.. autosummary::
   :toctree: generated

    run
    plot


Accessors
---------

.. :currentmodule:: fecsep.accessors

.. automodule:: fecsep.accessors

.. autosummary::
   :toctree: generated

    query_isc_gcmt
    from_zenodo
    from_git



Evaluations
-----------

Additional `pyCSEP` evaluations


.. :currentmodule:: fecsep.extra

.. automodule:: fecsep.extra

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

.. :currentmodule:: fecsep.utils

.. automodule:: fecsep.utils

.. autosummary::
   :toctree: generated

    parse_csep_func
    parse_timedelta_string
    read_time_config
    read_region_config
    time_windows_ti
    time_windows_td
    Task
    Task.run
    Task.check_exist
    timewindow_str
    plot_sequential_likelihood
    magnitude_vs_time



Database parsers
----------------

.. :currentmodule:: fecsep.readers

.. automodule:: fecsep.readers

.. autosummary::
   :toctree: generated

    ForecastParsers.dat
    ForecastParsers.xml
    ForecastParsers.quadtree
    ForecastParsers.csv
    ForecastParsers.hdf5
    HDF5Serializer.grid2hdf5
    serialize


