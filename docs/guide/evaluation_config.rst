.. _evaluation_config:

Evaluations Definition
======================

**floatCSEP** evaluate forecasts using the routines defined in **pyCSEP** (See `Testing Theory <https://docs.cseptesting.org/getting_started/theory.html>`_). Depending on the forecast types (e.g., **GriddedForecasts** or **CatalogForecasts**), different evaluation functions can be used.

Each evaluation specifies a ``func`` parameter, representing the evaluation function to be applied, a configuration of the function with ``func_kwargs`` (e.g., number of simulations, confidence intervals) and a ``plot_func`` parameter for visualizing the results. Evaluations for **GriddedForecasts** typically use functions from :mod:`csep.core.poisson_evaluations` or :mod:`csep.core.binomial_evaluations`, while evaluations for **CatalogForecasts** use functions from :mod:`csep.core.catalog_evaluations`.

.. important::

    An evaluation in ``test_config`` points to a **pyCSEP** `evaluation function <https://docs.cseptesting.org/concepts/evaluations.html>`_, valid for the forecast class.


**Example Configuration**:

.. code-block:: yaml
   :caption: test_config.yml

    - S-test:
        func: poisson_evaluations.spatial_test
        plot_func: plot_poisson_consistency_test
        plot_kwargs:
          one_sided_lower: True
    - T-test:
        func: poisson_evaluations.paired_t_test
        ref_model: Model A
        plot_func: plot_comparison_test


Evaluation Parameters
---------------------

Each evaluation listed in ``test_config`` accepts the following parameters:

.. list-table::
   :widths: 30 80
   :header-rows: 1

   * - **Parameter**
     - **Description**
   * - **func** (required)
     - Specify which evaluation/test function to run. Must be a **pyCSEP** ``{module}.{function}`` suite \
       (e.g., :func:`poisson_evaluations.number_test <csep.core.poisson_evaluations.number_test>`) or
       **floatCSEP** function.
   * - **func_kwargs**
     - Any keyword argument to control the specific **func**. For example, :func:`poisson_evaluations.spatial_test <csep.core.poisson_evaluations.spatial_test>` may be configured with ``num_simulations: 2000``.
   * - **plot_func** (required)
     - The function to plot the evaluation results, from either the :mod:`csep.utils.plots` module (e.g., :func:`plot_poisson_consistency_test <csep.utils.plots.plot_poisson_consistency_test>`) or **floatCSEP** :mod:`~floatcsep.utils.helpers` module.
   * - **plot_args**
     - Arguments passed to customize the plot function. Can be titles, labels, colors, font size, etc. Review the documentation of the respective function.
   * - **plot_kwargs**
     - Keyword arguments to customize the plot function. Review the documentation of the respective function.
   * - **ref_model**
     - A reference model against which the current model is compared in comparative tests (e.g., `Model A`).
   * - **markdown**
     - A description of the test to be used as caption when reporting results


Evaluations Functions
---------------------

**floatCSEP** supports the following evaluations:


.. dropdown:: **Evaluations for GriddedForecasts**
   :animate: fade-in-slide-down
   :icon: list-unordered

   .. list-table::
      :widths: 20 80
      :header-rows: 1

      * - **Function**
        - **Evaluates:**
      * - :func:`poisson_evaluations.number_test <csep.core.poisson_evaluations.number_test>`
        - the total number of forecasted events compared to the observed events using a Poisson distribution.
      * - :func:`poisson_evaluations.spatial_test <csep.core.poisson_evaluations.spatial_test>`
        - the forecasted spatial distribution relative to the observed events using a Poisson distribution.
      * - :func:`poisson_evaluations.magnitude_test <csep.core.poisson_evaluations.magnitude_test>`
        - the forecasted magnitude distribution relative to the observed events using a Poisson distribution.
      * - :func:`poisson_evaluations.conditional_likelihood_test <csep.core.poisson_evaluations.conditional_likelihood_test>`
        - the likelihood of the observed events given the forecasted rates, conditioned on the total earthquake occurrences, assuming a Poisson distribution.
      * - :func:`poisson_evaluations.paired_t_test <csep.core.poisson_evaluations.paired_t_test>`
        - the information gain between one forecast to a reference (``ref_model``), and test for a significant difference by using a paired T-test.
      * - :func:`binomial_evaluations.binary_spatial_test <csep.core.binomial_evaluations.binary_spatial_test>`
        - the forecasted spatial distribution relative to the observed events, assuming a Binary/Bernoulli process.
      * - :func:`binomial_evaluations.binary_likelihood_test <csep.core.binomial_evaluations.binary_likelihood_test>`
        - the likelihood of the observed events given the forecasted rates, assuming a Binary distribution.
      * - :func:`binomial_evaluations.negative_binomial_number_test <csep.core.binomial_evaluations.negative_binomial_number_test>`
        - the total number of forecasted events compared to the observed events using a Negative Binomial distribution.
      * - :func:`brier_score <floatcsep.utils.helpers.brier_score>`
        - the forecast skill using a quadratic metric rather than logarithmic. Does not penalize false-negatives as much as log-likelihood metrics.
      * - :func:`vector_poisson_t_w_test <floatcsep.utils.helpers.vector_poisson_t_w_test>`
        - a forecast skill compared to multiple forecasts, by carrying out the paired_t_test and w_test jointly.
      * - :func:`sequential_likelihood <floatcsep.utils.helpers.sequential_likelihood>`
        - the temporal evolution of log-likelihoods scores.
      * - :func:`sequential_information_gain <floatcsep.utils.helpers.sequential_information_gain>`
        - the temporal evolution of the information gain in time, compared to a ``ref_model``.



.. dropdown:: **Evaluations for CatalogForecasts**
   :animate: fade-in-slide-down
   :icon: list-unordered

   .. list-table::
      :widths: 20 80
      :header-rows: 1

      * - **Function**
        - **Evaluates:**
      * - :func:`catalog_evaluations.number_test <csep.core.catalog_evaluations.number_test>`
        - the total number of forecasted events compared to observed events in an earthquake catalog.
      * - :func:`catalog_evaluations.spatial_test <csep.core.catalog_evaluations.spatial_test>`
        - the spatial distribution of forecasted vs. observed earthquake events in an earthquake catalog.
      * - :func:`catalog_evaluations.magnitude_test <csep.core.catalog_evaluations.magnitude_test>`
        - the magnitude distribution of forecasted events to those observed in the earthquake catalog.
      * - :func:`catalog_evaluations.pseudolikelihood_test <csep.core.catalog_evaluations.pseudolikelihood_test>`
        - the pseudolikelihood of the observed events, given the forecasted synthetic catalogs
      * - :func:`catalog_evaluations.calibration_test <csep.core.catalog_evaluations.calibration_test>`
        - the consistency of multiple test-quantiles in time with the expected uniform distribution using a Kolmogorov-Smirnov test.

.. note::

   Check each function's `docstring` to see which ``func_kwargs`` are compatible with it.

Plotting Functions
------------------

**floatCSEP** supports the following:

.. dropdown::  Plotting functions
   :animate: fade-in-slide-down
   :icon: list-unordered

   .. list-table::
      :widths: 20 80
      :header-rows: 1

      * - **Plotting function**
        - **Compatible with:**
      * - :obj:`~csep.utils.plots.plot_poisson_consistency_test`
        - :func:`poisson_evaluations.number_test <csep.core.poisson_evaluations.number_test>`, :func:`poisson_evaluations.spatial_test <csep.core.poisson_evaluations.spatial_test>`, :func:`poisson_evaluations.magnitude_test <csep.core.poisson_evaluations.magnitude_test>`, :func:`poisson_evaluations.conditional_likelihood_test <csep.core.poisson_evaluations.conditional_likelihood_test>`.
      * - :obj:`~csep.utils.plots.plot_consistency_test`
        - :func:`binomial_evaluations.negative_binomial_number_test <csep.core.binomial_evaluations.negative_binomial_number_test>`, :func:`binomial_evaluations.binary_likelihood_test <csep.core.binomial_evaluations.binary_likelihood_test>`, :func:`binomial_evaluations.binary_spatial_test <csep.core.binomial_evaluations.binary_spatial_test>`, :func:`brier_score <floatcsep.utils.helpers.brier_score>`, :func:`catalog_evaluations.number_test <csep.core.catalog_evaluations.number_test>`, :func:`catalog_evaluations.magnitude_test <csep.core.catalog_evaluations.magnitude_test>`, :func:`catalog_evaluations.spatial_test <csep.core.catalog_evaluations.spatial_test>`, :func:`catalog_evaluations.pseudolikelihood_test <csep.core.catalog_evaluations.pseudolikelihood_test>`
      * - :obj:`~csep.utils.plots.plot_comparison_test`
        - :func:`poisson_evaluations.paired_t_test <csep.core.poisson_evaluations.paired_t_test>`
      * - :obj:`~csep.utils.plots.plot_number_test`
        - :func:`catalog_evaluations.number_test <csep.core.catalog_evaluations.number_test>`
      * - :obj:`~csep.utils.plots.plot_magnitude_test`
        - :func:`catalog_evaluations.magnitude_test <csep.core.catalog_evaluations.magnitude_test>`
      * - :obj:`~csep.utils.plots.plot_spatial_test`
        - :func:`catalog_evaluations.spatial_test <csep.core.catalog_evaluations.spatial_test>`
      * - :obj:`~csep.utils.plots.plot_likelihood_test`
        - :func:`catalog_evaluations.pseudolikelihood_test <csep.core.catalog_evaluations.pseudolikelihood_test>`
      * - :obj:`~csep.utils.plots.plot_calibration_test`
        - :func:`catalog_evaluations.calibration_test <csep.core.catalog_evaluations.calibration_test>`
      * - :obj:`~floatcsep.utils.helpers.plot_sequential_likelihood>`
        - :func:`sequential_likelihood <floatcsep.utils.helpers.sequential_likelihood>`, :func:`sequential_information_gain <floatcsep.utils.helpers.sequential_information_gain>`
      * - :obj:`~floatcsep.utils.helpers.plot_matrix_comparative_test`
        - :func:`vector_poisson_t_w_test <floatcsep.utils.helpers.vector_poisson_t_w_test>`

.. note::

   Check each plot functions's `docstring` to see which ``plot_args`` and ``plot_kwargs`` are compatible with it.



It is also possible to assign two or more plotting functions to a test, the ``plot_args`` and ``plot_kwargs`` of which can be placed as dictionaries indented beneath the functions:

**Example**:

.. code-block:: yaml
   :caption: test_config.yml

   - Number Test:
      func: catalog_evaluations.number_test
      plot_func:
         - plot_number_test:
               plot_args:
                  title: Number test distribution
         - plot_consistency_test:
               plot_args:
                  linewidth: 2
               plot_kwargs:
                  one_sided_lower: True
