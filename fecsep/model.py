import os
import git
from typing import List, Callable, Union
from datetime import datetime

from csep.core.forecasts import GriddedForecast
from csep.utils.time_utils import decimal_year

from fecsep.accessors import from_zenodo, from_git
from fecsep.readers import ForecastParsers, HDF5Serializer
from fecsep.utils import parse_csep_func, timewindow_str


class Model:
    """

    Class defining a forecast generating Model. Initializes a model source
    either from filesystem or web repositories, contains information about
    the Model's typology, and maps accordingly to a forecast generating
    function.

    Args:
        name (str): Name of the model
        path (str): Relative path of the model (file or runnable code)
                    to the Experiment's instance path
        forecast_unit (float): Temporal unit of the forecast. Default to
                               years in time-independent models and days
                               in time-dependent
        use_db (bool): Flag the use of a database for speed/memory purposes
        func (str, ~collections.abc.Callable): Forecast generating func (for code models)
        func_kwargs (dict): Arguments to pass into `func`
        zenodo_id (int): Zenodo ID or record of the Model
        giturl (float): Link to a git repository
        repo_hash (float): Specific commit/branch/tag hash.
        authors (list[str]): Authors' names metadata
        doi: Digital Object Identifier metadata:
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

    def __init__(self, name: str, path: str,
                 forecast_unit: float = 1, use_db: bool = False,
                 func: Union[str, Callable] = None, func_kwargs: dict = None,
                 zenodo_id: int = None, giturl: str = None,
                 repo_hash: str = None,
                 authors: List[str] = None,
                 doi: str = None) -> None:

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
            self.make_db()

        # INSTANTIATE ATTRIBUTES TO BE POPULATED IN runtime
        self.forecasts = {}

    @property
    def path(self) -> str:
        """
        (to be deprecated in following versions)

        Returns:
            The path pointing to the source file, or to the HDF5 database
         if this exists
        """
        if self.dbpath:
            return self.dbpath
        else:
            return self._path

    def get_source(self, zenodo_id: int = None, giturl: str = None,
                   **kwargs) -> None:
        """

        Search (or download/clone) the model source in the filesystem, zenodo
        and git. Identifies if the instance path points to a file or to its
        parent directory

        Args:
            zenodo_id (int): Zenodo identifier of the repository. Usually as
             `https://zenodo.org/record/{zenodo_id}`
            giturl (str): git remote repository URL from which to clone the
             source
            **kwargs: see :func:`~fecsep.utils.from_zenodo` and
             :func:`~fecsep.utils.from_git`


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

    def make_db(self, force: bool = False) -> None:
        """
        Seralizes a forecast file into a HDF5 file, widh identical name
        but different extension.

        Args:
            force (bool): Forces the serialization even if the DB already
             exists

        """
        # todo Think about distinction btwn 'TI' and 'Gridded' models.
        if self._class == 'ti':
            db_func = HDF5Serializer.grid2hdf5
            dbpath = os.path.splitext(self.path)[0] + '.hdf5'  # todo: reg
            if not os.path.isfile(dbpath) or force:
                db_func(self.path, self._fmt, dbpath)  # todo: reg
            self.dbpath = dbpath
            self._fmt = 'hdf5'
        else:
            raise NotImplementedError('TD serialization not implemented')

    def rm_db(self) -> None:
        """ Clean up the generated HDF5 File"""

        if os.path.isfile(self.dbpath):
            os.remove(self.dbpath)
        else:
            print("The HDF5 file does not exist")

    def create_forecast(self, start_date: datetime, end_date: datetime,
                        **kwargs) -> None:
        """

        Creates a forecast from the model source and a given time window

        Args:
            start_date (~datetime.datetime): Start of the forecast
            end_date (~datetime.datetime): End of the forecast
            **kwargs:

        """

        if self._src == 'file':
            self.forecast_from_file(start_date, end_date, **kwargs)

        else:
            # Forecasts are created from file
            self.make_forecast_ti(start_date, end_date, **kwargs)

    def forecast_from_func(self, start_date: datetime, end_date: datetime,
                           **kwargs) -> None:
        raise NotImplementedError('TBI for time-dependent models')

    def forecast_from_file(self, start_date: datetime, end_date: datetime,
                           **kwargs) -> None:
        """

        Generates a forecast from a file, by parsing and scaling it to
        the desired time window. Handles if the model or the forecast were
        already dumped to a DB previously.

        Args:
            start_date (~datetime.datetime): Start of the forecast
            end_date (~datetime.datetime): End of the forecast
            **kwargs: Keyword arguments for
             :class:`csep.core.forecasts.GriddedForecast`

        """

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
            end_time=end_date,
            **kwargs
        )

        forecast = forecast.scale(time_horizon / self.forecast_unit)
        print(
            f"Forecast expected count: {forecast.event_count:.2f}"
            f" with scaling parameter: {time_horizon:.1f}")

        self.forecasts[
            tstring] = forecast  # todo option to keep in memory or drop

    def get_forecast(self, start: datetime, end: datetime) -> None:
        """ Wrapper that just returns a forecast,
         hiding the processing (db storage, ti_td, etc.) under the hood"""
        pass

    def to_dict(self):
        # todo: modify this function to include more state from the class
        out = {}
        included = ['name', 'path']
        for k, v in self.__dict__.items():
            if k in included:
                out[k] = v
        return out

    @classmethod
    def from_dict(cls, record: dict):
        """
        Returns a Model instance from a dictionary containing the required
        atrributes. Can be used to quickly instantiate from a .yml file.

        Args:
            record (dict): Contains the keywords from the `__init__` method.

        Returns:
            A Model instance
        """

        if 'name' in record.keys():
            return cls(**record)
        elif len(record) != 1:
            raise IndexError('A single model has not been passed')
        name = next(iter(record))
        return cls(name=name, **record[name])
