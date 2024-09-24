.. _case_b:

B - Multiple Models and Tests
=============================

The following example is an experiment including **multiple** time-independent forecasts and **multiple** evaluations.

.. currentmodule:: floatcsep

.. admonition:: **TL; DR**

    In a terminal, navigate to ``floatcsep/tutorials/case_b`` and type:

    .. code-block:: console

        $ floatcsep run config.yml

    After the calculation is complete, the results will be summarized in ``results/report.md``.

.. contents:: Contents
    :local:
    :depth: 2


Experiment Components
---------------------

The source code can be found in the ``tutorials/case_b`` folder or in  `GitHub <https://github.com/cseptesting/floatcsep/blob/main/tutorials/case_b>`_. The input structure of the experiment is:

::

    case_b
        └── models
            ├── model_a.csv
            ├── model_b.csv
            ├── model_c.csv
            └── model_d.csv
        ├── config.yml
        ├── catalog.json
        ├── models.yml
        ├── tests.yml
        └── region.txt

.. important::
    Although not necessary, the testing catalog is here defined in the ``.json`` format, which is the default catalog used by ``floatcsep``, as it allows the storage of metadata.

.. note::
    A catalog can be stored as ``.json`` with :meth:`CSEPCatalog.write_json() <csep.core.catalogs.CSEPCatalog.write_json>` using ``pycsep``


Configuration
-------------

In this example, the time, region and catalog specifications are written in the ``config.yml`` file.

.. literalinclude:: ../../tutorials/case_b/config.yml
   :caption: tutorials/case_b/config.yml
   :language: yaml
   :lines: 3-15

whereas the models' and tests' configurations are referred to external files for better readability

.. literalinclude:: ../../tutorials/case_b/config.yml
   :language: yaml
   :lines: 17-18


Models
~~~~~~
    The model configuration is now set in the ``models.yml`` file, where a list of model names specify their file paths.

    .. literalinclude:: ../../tutorials/case_b/models.yml
       :caption: tutorials/case_b/models.yml
       :language: yaml

Evaluations
~~~~~~~~~~~
    The evaluations are defined in the ``tests.yml`` file as a list of evaluation names, with their  functions and plots (see :doc:`pycsep:concepts/evaluations`). In this example, we use the  N-, M-, S- and CL-consistency tests, along with the comparison T-test.

    .. literalinclude:: ../../tutorials/case_b/tests.yml
       :language: yaml
       :caption: tutorials/case_b/tests.yml

    .. note::
         Plotting keyword arguments can be set in the ``plot_kwargs`` option - see :func:`~csep.utils.plots.plot_poisson_consistency_test` and :func:`~csep.utils.plots.plot_comparison_test` -.

    .. important::
         Comparison tests (such as the ``paired_t_test``) requires a reference model, whose name should be set in ``ref_model`` at the given test configuration.

Running the experiment
----------------------


The experiment can be run by simply navigating to the ``tutorials/case_b`` folder in the terminal an type.

.. code-block:: console

    $ floatcsep run config.yml

This will automatically set all the file paths of the calculation (testing catalogs, evaluation results, figures) and will display a summarized report in ``results/report.md``.


