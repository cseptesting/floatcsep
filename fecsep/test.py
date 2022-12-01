import json

from csep.core.catalogs import CSEPCatalog

from fecsep.utils import parse_csep_func


class Test:
    """

    """
    _types = {'consistency': ['number_test', 'spatial_test', 'magnitude_test',
                              'likelihood_test', 'conditional_likelihood_test',
                              'negative_binomial_number_test',
                              'binary_spatial_test', 'binomial_spatial_test',
                              'brier_score',
                              'binary_conditional_likelihood_test'],
              'comparative': ['paired_t_test', 'w_test',
                              'binary_paired_t_test'],
              'fullcomp': ['vector_poisson_t_w_test'],
              'sequential': ['sequential_likelihood'],
              'seqcomp': ['sequential_information_gain']}

    def __init__(self, name, func, markdown='', func_args=None,
                 func_kwargs=None, plot_func=None,
                 plot_args=None, plot_kwargs=None, model=None, ref_model=None,
                 path=None):
        """

        :param name:
        :param func:
        :param func_args:
        :param plot_func:
        :param plot_args:
        :param ref_model:
        :param kwargs:
        """
        self.name = name
        self.func = parse_csep_func(func)
        self.func_kwargs = func_kwargs  # todo set default args from exp?
        self.func_args = func_args

        self.plot_func = parse_csep_func(plot_func)
        self.plot_args = plot_args or {}  # todo default args?
        self.plot_kwargs = plot_kwargs or {}
        self.ref_model = ref_model
        self.model = model
        self.path = path
        self.markdown = markdown

        self._type = None

    def compute(self,
                timewindow,
                catpath,
                model,
                path,
                ref_model=None,
                region=None):

        if self.type == 'comparative':
            forecast = model.forecasts[timewindow]
            catalog = CSEPCatalog.load_json(catpath)
            # todo: check region consistency between grid and qtree
            catalog.region = forecast.region
            ref_forecast = ref_model.forecasts[timewindow]
            test_args = (forecast, ref_forecast, catalog)

        elif self.type == 'fullcomp':
            ref_forecast = ref_model.forecasts[timewindow]
            catalog = CSEPCatalog.load_json(catpath)
            catalog.region = ref_forecast.region
            forecast_batch = [model_i.forecasts[timewindow] for model_i in
                              model]
            test_args = (ref_forecast, forecast_batch, catalog)

        elif self.type == 'sequential':
            forecasts = [model.forecasts[i] for i in timewindow]
            catalogs = [CSEPCatalog.load_json(i) for i in catpath]
            for i in catalogs:
                i.region = forecasts[0].region
            test_args = (forecasts, catalogs, timewindow)

        elif self.type == 'seqcomp':
            forecasts = [model.forecasts[i] for i in timewindow]
            ref_forecasts = [ref_model.forecasts[i] for i in timewindow]
            catalogs = [CSEPCatalog.load_json(i) for i in catpath]
            for i in catalogs:
                i.region = forecasts[0].region
            test_args = (forecasts, ref_forecasts, catalogs, timewindow)

        else:  # consistency
            forecast = model.forecasts[timewindow]
            catalog = CSEPCatalog.load_json(catpath)
            catalog.region = forecast.region
            test_args = (forecast, catalog)

        result = self.func(*test_args, **self.func_kwargs)

        with open(path, 'w') as _file:
            json.dump(result.to_dict(), _file, indent=4)
        # return test_args

    def to_dict(self):
        out = {}
        included = ['name', 'model', 'ref_model', 'path', 'func_kwargs']
        for k, v in self.__dict__.items():
            if k in included and v is not None:
                out[k] = v
        return out

    def __str__(self):
        return (
            f"name: {self.name}\n"
            f"reference model: {self.ref_model}\n"
            f"kwargs: {self.func_kwargs}\n"
            f"path: {self.path}"
        )

    @property
    def type(self):

        self._type = None
        for ty, funcs in Test._types.items():
            if self.func.__name__ in funcs:
                self._type = ty

        return self._type

    @classmethod
    def from_dict(cls, record):
        if len(record) != 1:
            raise IndexError('A single test has not been passed')
        name = next(iter(record))
        return cls(name=name, **record[name])
