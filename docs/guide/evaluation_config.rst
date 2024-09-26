.. _evaluation_config:

Evaluations Definition
======================

**floatCSEP** evaluate forecasts using the testing procedures from **pyCSEP** (See `Testing Theory <https://docs.cseptesting.org/getting_started/theory.html>`_). Depending on the forecast type (e.g., **GriddedForecasts** or **CatalogForecasts**), different evaluation functions can be used. T

Each evaluation specifies a `func` parameter, representing the evaluation function to be applied, and a `plot_func` parameter for visualizing the results.

Evaluations for **GriddedForecasts** typically use functions from :mod:`csep.core.poisson_evaluations` or :mod:`csep.core.binomial_evaluations`, while evaluations for **CatalogForecasts** use functions from :mod:`csep.core.catalog_evaluations`.

The structure of the evaluation configuration file is similar to the model configuration, with multiple tests, each pointing to a specific evaluation function and plotting method.

**Example Configuration**:

.. code-block:: yaml

    - N-test:
        func: poisson_evaluations.number_test
        plot_func: plot_poisson_consistency_test
    - S-test:
        func: poisson_evaluations.spatial_test
        plot_func: plot_poisson_consistency_test
        plot_kwargs:
          one_sided_lower: True
    - T-test:
        func: poisson_evaluations.paired_t_test
        ref_model: Model A
        plot_func: plot_comparison_test


Evaluation Parameters:
----------------------

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - **Parameter**
     - **Description**
   * - **func** (required)
     - The evaluation function, specifying which test to run. Must be an available function from the pyCSEP evaluation suite (e.g., `poisson_evaluations.number_test`).
   * - **plot_func** (required)
     - The function to plot the evaluation results, specified from the available plotting functions (e.g., `plot_poisson_consistency_test`).
   * - **plot_args**
     - Arguments passed to customize plot titles, labels, or font size.
   * - **plot_kwargs**
     - Keyword arguments passed to the plotting function for fine-tuning plot appearance (e.g., `one_sided_lower: True`).
   * - **ref_model**
     - A reference model against which the current model is compared in comparative tests (e.g., `Model A`).
   * - **markdown**
     - A description of the test to be used as caption when reporting results


Evaluations Functions:
----------------------

Depending on the type of forecast being evaluated, different evaluation functions are used:

1. **GriddedForecasts**:

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - **Function**
     - **Description**
   * - **poisson_evaluations.number_test**
     - Evaluates the forecast by comparing the total number of forecasted events with the observed events using a Poisson distribution.
   * - **poisson_evaluations.spatial_test**
     - Compares the spatial distribution of forecasted events to the observed events.
   * - **poisson_evaluations.magnitude_test**
     - Evaluates the forecast by comparing the magnitude distribution of forecasted events with observed events.
   * - **poisson_evaluations.conditional_likelihood_test**
     - Tests the likelihood of observed events given the forecasted rates, conditioned on the total earthquake occurrences.
   * - **poisson_evaluations.paired_t_test**
     - Calculate the information gain between one forecast to a reference (``ref_model``), and test a significant difference by using a paired T-test.
   * - **binomial_evaluations.binary_spatial_test**
     - Binary spatial test to compare forecasted and observed event distributions.
   * - **binomial_evaluations.binary_likelihood_test**
     - Likelihood test likelihood of observed events given the forecasted rates, assuming a Binary distribution
   * - **binomial_evaluations.negative_binomial_number_test**
     - Evaluates the number of events using a negative binomial distribution, comparing observed and forecasted event counts.
   * - **brier_score**
     - Uses a quadratic metric rather than logarithmic. Does not penalize false-negatives as much as log-likelihood metrics
   * - **vector_poisson_t_w_test**
     - Carries out the paired_t_test and w_test for a single forecast compared to multiple.
   * - **sequential_likelihood**
     - Obtain the distribution of log-likelihoods in time.
   * - **sequential_information_gain**
     - Obtain the distribution of information gain in time, compared to a ``ref_model``.


2. **CatalogForecasts**:



