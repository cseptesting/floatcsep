C - Multiple Time Windows
=========================

.. currentmodule:: floatcsep

.. contents::
    :local:

.. admonition:: **TL; DR**

    In a terminal, navigate to ``floatcsep/examples/case_c`` and type:

    .. code-block:: console

        $ floatcsep run config.yml

    After the calculation is complete, the results will be summarized in ``results/report.md``.


Artifacts
---------

The following example shows an experiment with multiple time windows. The input structure of the experiment is:

::

    case_c
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

Configuration
-------------

Time
~~~~

    The time configuration now set the time intervals between the start and end dates.

    .. literalinclude:: ../../examples/case_c/config.yml
       :language: yaml
       :lines: 3-7

    .. note::

        The time interval ``growth`` can be either ``cumulative`` (all windows start from ``start_date``) or ``incremental`` (each window starts from the previous window's end).

    The results of the experiment run will be associated with each time window (``2010-01-01_2011-01-01``, ``2010-01-01_2012-01-01``, ``2010-01-01_2013-01-01``, ...).



Evaluations
~~~~~~~~~~~
    The experiment's evaluations are defined in ``tests.yml``, which can now include temporal evaluations (see :func:`~floatcsep.extras.sequential_likelihood`, :func:`~floatcsep.extras.sequential_information_gain`, :func:`~floatcsep.utils.plot_sequential_likelihood`).

    .. literalinclude:: ../../examples/case_c/tests.yml
       :language: yaml

    .. note::

        Plot arguments (title, labels, font sizes, axes limits, etc.) can be passed as a dictionary in ``plot_args`` (see details in :func:`~csep.utils.plots.plot_poisson_consistency_test`)

Results
-------

The :obj:`~floatcsep.cmd.main.run` command creates the result path tree for all time windows.

*  The testing catalog of the window is stored in ``results/{window}/catalog``  in ``json`` format. This is a subset of the global testing catalog.
*  Human-readable results are found in ``results/{window}/evaluations``
*  Catalog and evaluation results figures in ``results/{window}/figures``.
*  The complete results are summarized in ``results/report.md``

The report now shows the temporal evaluations for all time-windows, whereas the discrete evaluations are shown only for the last time window.


