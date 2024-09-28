.. _case_c:

C - Multiple Time Windows
=========================

The following example shows an experiment with **multiple time windows**.

.. currentmodule:: floatcsep

.. admonition:: **TL; DR**

    In a terminal, navigate to ``floatcsep/tutorials/case_c`` and type:

    .. code-block:: console

        $ floatcsep run config.yml

    After the calculation is complete, the results will be summarized in ``results/report.md``.

.. contents:: Contents
    :local:



Experiment Components
---------------------

The source code can be found in the ``tutorials/case_c`` folder or in  `GitHub <https://github.com/cseptesting/floatcsep/blob/main/tutorials/case_c>`_. The input structure of the experiment is:

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

    The time configuration now sets a sequence of time intervals between the start and end dates.

    .. literalinclude:: ../../tutorials/case_c/config.yml
        :caption: tutorials/case_c/config.yml
        :language: yaml
        :lines: 3-7

    .. note::

        The time interval ``growth`` can be either ``cumulative`` (all windows start from ``start_date``) or ``incremental`` (each window starts from the previous window's end).

    The results of the experiment run will be associated with each time window (``2010-01-01_2011-01-01``, ``2010-01-01_2012-01-01``, ``2010-01-01_2013-01-01``, ...).



Evaluations
~~~~~~~~~~~
    The experiment's evaluations are defined in ``tests.yml``, which can now include temporal evaluations (see :obj:`~floatcsep.utils.helpers.sequential_likelihood`, :obj:`~floatcsep.utils.helpers.sequential_information_gain`, :obj:`~floatcsep.utils.helpers.plot_sequential_likelihood`).

    .. literalinclude:: ../../tutorials/case_c/tests.yml
        :language: yaml
        :caption: tutorials/case_c/tests.yml

    .. note::

        Plot arguments (title, labels, font sizes, axes limits, etc.) can be passed as a dictionary in ``plot_args`` (see the arguments details in :func:`~csep.utils.plots.plot_poisson_consistency_test`)

Results
-------

The :obj:`~floatcsep.commands.main.run` command

.. code-block:: console

    $ floatcsep run config.yml

now creates the result path tree for all time windows.

*  The testing catalog of the window is stored in ``results/{time_window}/catalog``  in ``json`` format. This is a subset of the global testing catalog.
*  Human-readable results are found in ``results/{time_window}/evaluations``
*  Catalog and evaluation results figures in ``results/{time_window}/figures``.
*  The complete results are summarized in ``results/report.md``

The report shows the temporal evaluations for all time-windows, whereas the discrete evaluations are shown only for the last time window.


