Experiment B - Multiple Models and Tests
==========================================

.. currentmodule:: fecsep

.. contents::


.. admonition:: **TL; DR**

    In a terminal, navigate to ``fecsep/examples/case_b`` and type:

    .. code-block:: shell

        $ fecsep run config.yml

    After the calculation is complete, the results will be summarized in ``results/report.md``.


Experiment's artifacts
----------------------

The following example is an experiment including multiple forecasts and evaluations. The input structure of the experiment is:

::

    case_b
        └──  models
            ├── model_a.csv
            ├── model_b.csv
            ├── model_c.csv
            └── model_d.csv
        ├── config.yml
        ├── catalog.json
        ├── models.yml
        ├── tests.yml
        └── region.txt


The testing catalog is defined in ``json`` format, which is the default catalog used by ``fecsep``, as it allows the storage of metadata.

.. note::
    Using ``pycsep`` a catalog can be saved using :meth:`CSEPCatalog.write_json() <csep.core.catalogs.CSEPCatalog.write_json>`


Configuration
-------------

In this example, the time, region and catalog specifications are written in the ``config.yml`` file.

.. literalinclude:: ../../examples/case_b/config.yml
   :language: yaml
   :lines: 3-15

whereas the models' and tests' configurations are referred to external files for readability

.. literalinclude:: ../../examples/case_b/config.yml
   :language: yaml
   :lines: 17-18


Models
~~~~~~
The model configuration is now set in the ``models.yml`` file, where a list of model names specify their file paths.

.. literalinclude:: ../../examples/case_b/models.yml
   :language: yaml

Evaluations
~~~~~~~~~~~
The evaluations are defined in the ``tests.yml`` file as a list of evaluation names, with their  functions and plots (see :doc:`pycsep:concepts/evaluations`). In this example, we use the  N-, M-, S- and CL-consistency tests, along with the comparison T-test.

.. literalinclude:: ../../examples/case_b/tests.yml
   :language: yaml

.. note::
     Keyword arguments can be passed to the plot functions (see :func:`~csep.utils.plots.plot_poisson_consistency_test` and :func:`~csep.utils.plots.plot_comparison_test`).

.. note::
     Comparison tests (such as the ``paired_t_test``) requires a reference model, whose name should be set in ``ref_model`` at the given test configuration.

Running the experiment
----------------------

Run command
~~~~~~~~~~~

The experiment can be run by simply navigating to the ``examples/case_b`` folder in the terminal an type.

.. code-block:: shell

    fecsep run config.yml

This will automatically set all the file paths of the calculation (testing catalogs, evaluation results, figures) and will display a summarized report in ``results/report.md``.


Advanced
--------

The experiment run logic can be seen in the file ``case_b.py``, which executes the same example but in python source code. The run logic of the terminal commands ``run``, ``plot`` and ``reproduce`` can be seen in :class:`fecsep.cmd.main`


