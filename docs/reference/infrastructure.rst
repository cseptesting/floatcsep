Infrastructure Module
=====================

Here are shown the modules that manage the relations between the core classes of ``floatCSEP``
and the required workflow to run an Experiment.

Registries
----------

.. autoclass:: floatcsep.infrastructure.registries.ForecastRegistry
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: floatcsep.infrastructure.registries.ExperimentRegistry
   :members:
   :undoc-members:
   :show-inheritance:


Repositories
------------

.. autoclass:: floatcsep.infrastructure.repositories.CatalogForecastRepository
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: floatcsep.infrastructure.repositories.GriddedForecastRepository
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: floatcsep.infrastructure.repositories.ResultsRepository
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: floatcsep.infrastructure.repositories.CatalogRepository
   :members:
   :undoc-members:
   :show-inheritance:

Environments
------------

.. autoclass:: floatcsep.infrastructure.environments.CondaManager
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: floatcsep.infrastructure.environments.VenvManager
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: floatcsep.infrastructure.environments.DockerManager
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: floatcsep.infrastructure.environments.EnvironmentFactory
   :members:
   :undoc-members:
   :show-inheritance:


Engine
------

The components here are in charge of managing and executing the ``floatCSEP`` workflow.

.. autoclass:: floatcsep.infrastructure.engine.Task
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

.. autoclass:: floatcsep.infrastructure.engine.TaskGraph
   :members:
   :undoc-members:
   :show-inheritance:
