API Reference
=============

This contains a reference document to the feCSEP API.


Experiment
----------

.. :currentmodule:: fecsep.core

.. automodule:: fecsep.core


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


A model is defined using the :class:`Model` class.

.. autosummary::
   :toctree: generated

    Model
    Model.get_source
    Model.make_db
    Model.rm_db
    Model.create_forecast
    Model.make_forecast_td
    Model.make_forecast_ti
    Model.to_dict
    Model.from_dict


Test
----


A test is defined using the :class:`Test` class.

.. autosummary::
   :toctree: generated

    Test
    Test.compute
    Test.to_dict
    Test.from_dict




Running an Experiment
----------------------

.. :currentmodule:: fecsep.main

.. automodule:: fecsep.main

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
    sequential_likelihood
    plot_sequential_likelihood
    magnitude_vs_time



Database parsers
----------------

.. :currentmodule:: fecsep.dbparser

.. automodule:: fecsep.dbparser

.. autosummary::
   :toctree: generated

    HDF5Serializer.quadtree
    HDF5Serializer.dat
    HDF5Serializer.csv
    HDF5Serializer.xml
    load_from_hdf5


