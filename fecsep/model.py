import os
import git

from csep.core.forecasts import GriddedForecast
from csep.utils.time_utils import decimal_year

from fecsep.accessors import from_zenodo, from_git
from fecsep.readers import ForecastParsers, HDF5Serializer
from fecsep.utils import parse_csep_func, timewindow_str


class Model:
    def __init__(self, name, path,
                 forecast_unit=1,
                 authors=None,
                 doi=None,
                 use_db=False,
                 func=None,
                 func_kwargs=None,
                 zenodo_id=None,
                 giturl=None,
                 repo_hash=None):
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
                (ti - code)  >>> TBI, but probably out of the scope.
                
            (td - code - local)
            (td - code - git)
            (td - code - zenodo)
            
        Get forecasts:
            - FILE
                A - read from file, scale in runtime
                B - drop to db, scale from function in runtime   (only makes sense to speed things)
                C - drop to db, scale and drop to db
            - SOURCE
                D - run, read from file              (D similar to A) 
                E - run, store in db, read from db   (E similar to C)
    
        OPTIONS:
            - TI <-> FROM FILE     - SCALE IN RT
                                   - SCALE and drop to DB
            - TD <-> FROM FILE     - TO DB 
                     FROM SRC      - TO DB
                                   
        '''

        # todo list
        #  - Check format
        #  - Instantiate from source code
        #  - Check contents when instantiating from_git

        # INIT ATTRIBUTES
        self.name = name
        self.zenodo_id = zenodo_id
        self.giturl = giturl
        self.repo_hash = repo_hash
        self.authors = authors
        self.doi = doi
        self.forecast_unit = forecast_unit
        self.func = func
        self.func_args = func_kwargs
        self.use_db = use_db

        # SET MODEL CLASS
        if self.func:
            self._class = 'td'  # Time-Dependent todo: implement for TI
        else:
            self._class = 'ti'  # Time-Independent

        # PATHS # todo: get from registry
        self._path = path
        self._dir = None
        self.dbpath = None

        # MODEL SELF-DISCOVER ATTRS
        self._src = None
        self._fmt = None
        self.get_source(zenodo_id, giturl)

        if self.use_db:
            if self._class == 'ti':
                self.db_func = HDF5Serializer.grid2hdf5
                self.make_db()
            else:
                raise NotImplementedError('missing for Time-Dep')
        else:
            self.db_func = None

        # INSTANTIATE ATTRIBUTES TO BE POPULATED IN runtime
        self.forecasts = {}

    @property
    def path(self):
        if self.dbpath:
            return self.dbpath
        else:
            return self._path

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

        if bool(ext):  # model is a file
            self._dir = os.path.dirname(self.path)  # todo reg
            self._fmt = ext.split('.')[-1]
            self._src = 'file'
        else:  # model is bin
            self._dir = self.path  # todo reg
            self._src = 'bin'

        # Folder nor file exists -> get from zenodo or git
        if not os.path.exists(self.path):  # todo reg
            if zenodo_id is None and giturl is None:
                raise FileNotFoundError(
                    f"Model file or directory '{self.path}' not found")  # todo reg

            os.makedirs(self._dir, exist_ok=True)
            try:
                # Zenodo is the first source of retrieval
                from_zenodo(zenodo_id, self._dir, **kwargs)  # todo reg
            except KeyError or TypeError as zerror_:
                try:
                    from_git(giturl, self._dir, **kwargs)  # todo reg
                except (git.NoSuchPathError, git.CommandError) as giterror_:
                    if giturl is None:
                        raise KeyError('Zenodo identifier is not valid')
                    else:
                        raise git.NoSuchPathError('git url was not found')

            # Check if file or directory exists after downloading
            if bool(ext):
                path_exists = os.path.isfile(self.path)  # todo reg
            else:
                path_exists = os.path.isdir(self.path)  # todo reg

            assert path_exists

    def make_db(self, force=False):
        """

        Returns:

        """

        dbpath = os.path.splitext(self.path)[0] + '.hdf5'  # todo: reg
        if not os.path.isfile(dbpath) or force:
            self.db_func(self.path, self._fmt, dbpath)  # todo: reg
        self.dbpath = dbpath
        self._fmt = 'hdf5'

    def rm_db(self):

        if os.path.isfile(self.dbpath):
            os.remove(self.dbpath)
        else:
            print("The HDF5 file does not exist")

    def create_forecast(self, start_date, end_date, **kwargs):
        """
        Creates a forecast from a model and a time window
        :param start_date: A model configuration dict
        :param test_date: A test date to calculate the horizon
        :return: A pycsep.core.forecasts.GriddedForecast object
        """

        if self._src == 'file':
            self.forecast_from_file(start_date, end_date, **kwargs)

        else:
            # Forecasts are created from file
            self.make_forecast_ti(start_date, end_date, **kwargs)

    def forecast_from_func(self, start_date, end_date, **kwargs):
        pass

    def forecast_from_file(self, start_date, end_date, **kwargs):

        time_horizon = decimal_year(end_date) - \
                       decimal_year(start_date)
        tstring = timewindow_str([start_date, end_date])  # reg

        f_parser = getattr(ForecastParsers, self._fmt)
        rates, region, mags = f_parser(self.path)

        forecast = GriddedForecast(
            name=f'{self.name}',
            data=rates,
            region=region,
            magnitudes=mags,
            start_time=start_date,
            end_time=end_date
        )

        forecast = forecast.scale(time_horizon / self.forecast_unit)
        print(
            f"Forecast expected count: {forecast.event_count:.2f}"
            f" with scaling parameter: {time_horizon:.1f}")

        self.forecasts[
            tstring] = forecast  # todo option to keep in memory or drop

    def to_dict(self):
        # todo: modify this function to include more state from the class
        out = {}
        included = ['name', 'path']
        for k, v in self.__dict__.items():
            if k in included:
                out[k] = v
        return out

    def get_forecast(self, start, end):
        """ Wrapper that just returns a forecast,
         hiding the processing (db storage, ti_td, etc, etc) under the hood"""
        pass

    @classmethod
    def from_dict(cls, record):
        if 'name' in record.keys():
            return cls(**record)
        elif len(record) != 1:
            raise IndexError('A single model has not been passed')
        name = next(iter(record))
        return cls(name=name, **record[name])
