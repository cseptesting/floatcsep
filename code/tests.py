def matrix_poisson_t_test(forecasts, catalog, **kwargs):
    """ Computes Student's t-test for the information gain per earthquake over a list of forecasts
        
        Uses unique combinations of individual forecasts to perform pair-wise t-tests against all forecasts 
        provided to the function. 

        Args:
            forecasts (list of csep.GriddedForecast): list of forecasts to evaluate
            catalog (csep.AbstractBaseCatalog): evaluation catalog filtered consistent with forecast
            **kwargs: additional default arguments

        Returns:
            results (list of csep.EvaluationResult): iterable of evaluation results
    """

def negative_binomial_number_test(forecast, catalog):
    pass


def binomial_conditional_likelihood_test(forecast, catalog, **kwargs):
    pass


def binomial_conditional_likelihood_test(forecast, catalog, **kwargs):
    pass


def binomial_spatial_test(forecast, catalog, **kwargs):
    pass


def matrix_binary_t_test(forecasts, catalog, **kwargs):
    pass
