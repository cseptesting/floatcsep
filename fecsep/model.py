import os
import git

from csep.core.forecasts import GriddedForecast
from csep.utils.time_utils import decimal_year

from fecsep.accessors import from_zenodo, from_git
from fecsep.dbparser import load_from_hdf5
from fecsep.utils import parse_csep_func, timewindow_str


class Model:
    def __init__(self, name, path,
                 forecast_unit=1,
                 authors=None, doi=None,
                 func=None, func_args=None,
                 zenodo_id=None, giturl=None, repo_hash=None):
        """

        Args:
            name (str):
            path (str):
            forecast_unit (str):
            authors (str):
            doi (str):
            func:
            func_args:
            zenodo_id:
            giturl:
            repo_hash:
        """

        '''
        Model typologies:
        
            Updating
            - Time-Independent
            - Time-Dependent
            Origin
            - File
            - Code
            Source 
            - Local
            - Zenodo
            - Git
    
        To implement in beta version
            (ti - file - local)
            (ti - file - zenodo)
            (ti - file - git)
            
            (td - code - local)
            (td - code - git)
        '''

        # todo list
        #  - Check format
        #  - Instantiate from source code
        #  - Check contents when instantiating from_git

        self.name = name
        self.path = path
        self._dir = None
        self.dbpath = None

        self.zenodo_id = zenodo_id
        self.giturl = giturl
        self.repo_hash = repo_hash

        self.format = None
        self.get_source(zenodo_id, giturl)

        if self.format != 'src':
            self.dbserializer = parse_csep_func(self.format)
            self.make_db()

        self.authors = authors
        self.doi = doi

        self.forecast_unit = forecast_unit

        self.forecasts = {}
        self.func = func
        self.func_args = func_args

    def get_source(self, zenodo_id=None, giturl=None, **kwargs):
        """

        Search(and download/clone) the model source in the filesystem, zenodo
        and git. Identifies if the instance path points to a file or to its
        parent directory

        Args:
            zenodo_id: Zenodo identifier of the repository. Usually as
             `https://zenodo.org/record/{zenodo_id}`
            giturl: git remote repository URL from which to clone the source
            **kwargs: see :func:`~fecsep.utils.from_zenodo` and
             :func:`~fecsep.utils.from_git`

        Returns:

        """

        # Check if the provided path is a file or dir.
        ext = os.path.splitext(self.path)[-1]

        if bool(ext):
            self._dir = os.path.dirname(self.path)
            self.format = ext.split('.')[-1]
        else:
            self._dir = self.path
            self.format = 'src'

        if not os.path.exists(self.path):
            # It does not exist, get from zenodo or git
            if zenodo_id is None and giturl is None:
                raise FileNotFoundError(
                    f"Model file or directory '{self.path}' not found")
            # Model needs to be downloaded from zenodo/git
            os.makedirs(self._dir, exist_ok=True)
            try:
                # Zenodo is the first source of retrieval
                from_zenodo(zenodo_id, self._dir, **kwargs)
            except KeyError or TypeError as zerror_:
                try:
                    from_git(giturl, self._dir, **kwargs)
                except (git.NoSuchPathError, git.CommandError) as giterror_:
                    if giturl is None:
                        raise KeyError('Zenodo identifier is not valid')
                    else:
                        raise git.NoSuchPathError('git url was not found')

            # Check if file or directory exists after downloading
            if bool(ext):
                path_exists = os.path.isfile(self.path)
            else:
                path_exists = os.path.isdir(self.path)

            assert path_exists

    def make_db(self):
        """

        Returns:

        """

        self.dbpath = os.path.splitext(self.path)[0] + '.hdf5'

        if not os.path.isfile(self.dbpath):
            self.dbserializer(self.path, self.dbpath)

    def rm_db(self):

        if os.path.isfile(self.dbpath):
            os.remove(self.dbpath)
            return True
        else:
            print("The HDF5 file does not exist")
            return False

    def create_forecast(self, start_date, end_date, **kwargs):
        """
        Creates a forecast from a model and a time window
        :param start_date: A model configuration dict
        :param test_date: A test date to calculate the horizon
        :return: A pycsep.core.forecasts.GriddedForecast object
        """

        if self.path == self._dir:
            # Forecasts are created from source code
            self.make_forecast_td(start_date, end_date, **kwargs)
        else:
            # Forecasts are created from file
            self.make_forecast_ti(start_date, end_date, **kwargs)

    def make_forecast_td(self, start_date, end_date, **kwargs):
        pass

    def make_forecast_ti(self, start_date, end_date, **kwargs):

        time_horizon = decimal_year(end_date) - decimal_year(start_date)
        tstring = timewindow_str([start_date, end_date])

        # todo implement these functions in dbparser

        rates, region, magnitudes = load_from_hdf5(self.dbpath)

        forecast = GriddedForecast(
            name=f'{self.name}',
            data=rates,
            region=region,
            magnitudes=magnitudes,
            start_time=start_date,
            end_time=end_date
        )

        forecast = forecast.scale(time_horizon / self.forecast_unit)
        print(
            f"Forecast expected count: {forecast.event_count:.2f}"
            f" with scaling parameter: {time_horizon:.1f}")
        self.forecasts[tstring] = forecast

    def to_dict(self):
        # todo: modify this function to include more state from the class
        out = {}
        included = ['name', 'path']
        for k, v in self.__dict__.items():
            if k in included:
                out[k] = v
        return out

    @classmethod
    def from_dict(cls, record):
        if len(record) != 1:
            raise IndexError('A single model has not been passed')
        name = next(iter(record))
        return cls(name=name, **record[name])
