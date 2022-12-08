import os
import git
from typing import List, Callable, Union
from datetime import datetime

from csep.core.forecasts import GriddedForecast, CatalogForecast
from csep.utils.time_utils import decimal_year

from fecsep.accessors import from_zenodo, from_git
from fecsep.readers import ForecastParsers, HDF5Serializer, check_format
from fecsep.utils import timewindow2str, str2timewindow
from fecsep.registry import register


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
        func (str, ~collections.abc.Callable): Forecast generating function
                                               (for code models)
        func_kwargs (dict): Arguments to pass into `func`
        zenodo_id (int): Zenodo ID or record of the Model
        giturl (str): Link to a git repository
        repo_hash (str): Specific commit/branch/tag hash.
        authors (list[str]): Authors' names metadata
        doi: Digital Object Identifier metadata:

    """

    '''
    Model characteristics:
        Forecast:   - Gridded
                    - Catalog
        Updating:   - Time-Independent
                    - Time-Dependent
        Source:     - File
                    - Code
    Model typology:
    
    To implement in beta version:       
        - (grid - ti - file): e.g. CSEP1 style gridded forecasts
    To implement in further versions:
        - (grid - ti - code):  e.g. smoothed-seismicity model
        - (grid - td - code): e.g. EEPAS, STEP, Italy-NZ OEF models
        - (cat - td - code): e.g. ETAS model code
        - (cat - td - file): e.g OEF-ready Forecasts
        - (grid - td - file): e.g OEF-ready Forecasts

    Get forecasts options:
        - FILE   - read from file, scale in runtime             
                 - drop to db, scale from function in runtime   
        - CODE  - run, read from file              
                - run, store in db, read from db   

    '''

    @register
    def __init__(self, name: str, path: str,
                 forecast_unit: float = 1, use_db: bool = False,
                 func: Union[str, Callable] = None, func_kwargs: dict = None,
                 zenodo_id: int = None, giturl: str = None,
                 repo_hash: str = None, authors: List[str] = None,
                 doi: str = None) -> None:

        # todo:
        #  - Instantiate from source code

        # Initialize path
        self.path = path

        # Instantiate attributes
        self.name = name
        self.zenodo_id = zenodo_id
        self.giturl = giturl
        self.repo_hash = repo_hash
        self.authors = authors
        self.doi = doi
        self.forecast_unit = forecast_unit
        self.func = func
        self.func_kwargs = func_kwargs
        self.use_db = use_db

        # Set model temporal class default
        if self.func:
            self._class = 'td'  # Time-Dependent todo: implement for TI
        else:
            self._class = 'ti'  # Time-Independent

        # Instantiate attributes to be filled in run-time
        self.forecasts = {}

    def __getattr__(self, name) -> object:
        """ If fails to get an attribute, tries to get from Registry """

        if name != 'reg' and hasattr(self, 'reg'):
            try:
                return getattr(self.reg, name)
            except AttributeError:
                raise AttributeError(
                    f"{self.__class__.__name__} object named "
                    f"nor its Registry, has attribute {name}")
        else:
            raise AttributeError(
                f"'{self.__class__.__name__} 'object has no "
                f"attribute '{name}'")

    @property
    def path(self) -> str:
        """

        Returns:
            Path pointing to the source file, the HDF5 database, or src ode.
        """
        if hasattr(self, 'reg'):
            return self.reg.path
        else:
            return self._path

    @path.setter
    def path(self, new_path) -> None:
        """ Path setter. Original path passed when instantiated, is found at
         self.reg.meta['path']"""

        if hasattr(self, 'reg'):
            self.reg.path = new_path
        else:
            self._path = new_path

    def stage(self) -> None:
        """
        Pre-steps to make the model runnable before integrating
            - Get from filesystem, Zenodo or Git
            - Pre-check model fileformat
            - Initialize database
            - Run model quality assurance (unit tests, runnable from fecsep)
        """

        self.get_source(self.zenodo_id, self.giturl)
        check_format(self.path, self.fmt, self.func)
        if self.use_db:
            self.init_db()
        self.model_qa()

    def get_source(self, zenodo_id: int = None, giturl: str = None,
                   force: bool = False, **kwargs) -> None:
        """

        Search, download or clone the model source in the filesystem, zenodo
        and git, respectively. Identifies if the instance path points to a file
        or to its parent directory

        Args:
            zenodo_id (int): Zenodo identifier of the repository. Usually as
             `https://zenodo.org/record/{zenodo_id}`
            giturl (str): git remote repository URL from which to clone the
             source
            force (bool): Forces to re-query the model from a web repository
            **kwargs: see :func:`~fecsep.utils.from_zenodo` and
             :func:`~fecsep.utils.from_git`


        """

        if os.path.exists(self.path) and not force:
            return None
        else:
            if zenodo_id is None and giturl is None:
                raise FileNotFoundError(
                    f"Model file or directory '{self.path}' not found")

        os.makedirs(self.dir, exist_ok=True)
        try:
            # Zenodo is the first source of retrieval
            from_zenodo(zenodo_id, self.dir, force=force)
        except (KeyError, TypeError):
            try:
                from_git(giturl, self.dir, **kwargs)

            except (git.NoSuchPathError, git.CommandError):
                if giturl is None:
                    raise KeyError('Zenodo identifier is not valid')
                else:
                    raise git.NoSuchPathError('git url was not found')

        if not os.path.exists(self.dir) or not os.path.exists(self.path):
            raise FileNotFoundError(
                f"Directory '{self.dir}' or file {self.path}' do not exist. "
                f"Please check the specified 'path' matches the repo "
                f"structure")

    def init_db(self, dbpath: str = '', force: bool = False) -> None:
        """
        Initializes the database if `use_db` is True. If the model source is a
        file, seralizes the forecast into a HDF5 file. If source is a
        generating function or code, creates an empty DB

        Args:
            dbpath (str): Path to drop the HDF5 database. Defaults to same path
             replaced with an `hdf5` extension
            force (bool): Forces the serialization even if the DB already
             exists

        """
        # todo Think about distinction btwn 'TI' and 'Gridded' models.
        if self.fmt and self._class == 'ti':

            parser = getattr(ForecastParsers, self.fmt)
            rates, region, mag = parser(self.path)

            db_func = HDF5Serializer.grid2hdf5
            if not dbpath:
                dbpath = self.path.replace(self.fmt, 'hdf5')

            if not os.path.isfile(dbpath) or force:
                # Drop Source file into DB
                db_func(rates, region, mag,
                        hdf5_filename=dbpath,
                        unit=self.forecast_unit)

            self.path = dbpath

        else:
            raise NotImplementedError('TD serialization not implemented')

    def rm_db(self) -> None:
        """ Clean up the generated HDF5 File"""

        if self.use_db:
            if os.path.isfile(self.path) and self.fmt == 'hdf5':
                os.remove(self.path)
            else:
                print("The HDF5 file does not exist")

    def model_qa(self) -> None:
        """ Run model quality assurance (Unit and Integration tests).
        Should not run if the repo has not been modified.
        Not implemented """
        pass

    def get_forecast(self,
                     tstring: str = None,
                     start_date: datetime = None,
                     end_date: datetime = None
                     ) -> Union[GriddedForecast, CatalogForecast]:
        """ Wrapper that just returns a forecast, hiding the processing
        (db storage, ti_td, etc.) under the hood"""
        if tstring:
            start_date, end_date = str2timewindow(tstring)
        else:
            tstring = timewindow2str([start_date, end_date])

        if tstring in self.forecasts.keys():
            return self.forecasts[tstring]
        else:
            self.create_forecast(start_date, end_date)
            return self.forecasts[tstring]

    def create_forecast(self, start_date: datetime, end_date: datetime,
                        **kwargs) -> None:
        """

        Creates a forecast from the model source and a given time window

        Args:
            start_date (~datetime.datetime): Start of the forecast
            end_date (~datetime.datetime): End of the forecast
            **kwargs:

        """

        # Model src is a file
        if self.fmt:
            self.forecast_from_file(start_date, end_date, **kwargs)
        # Model src is a func or binary
        else:
            self.forecast_from_func(start_date, end_date, **kwargs)

    def forecast_from_file(self, start_date: datetime, end_date: datetime,
                           **kwargs) -> None:
        """

        Generates a forecast from a file, by parsing and scaling it to
        the desired time window. H

        Args:
            start_date (~datetime.datetime): Start of the forecast
            end_date (~datetime.datetime): End of the forecast
            **kwargs: Keyword arguments for
             :class:`csep.core.forecasts.GriddedForecast`

        """

        time_horizon = decimal_year(end_date) - decimal_year(start_date)
        tstring = timewindow2str([start_date, end_date])

        f_parser = getattr(ForecastParsers, self.fmt)
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

        scale = time_horizon / self.forecast_unit
        if scale != 1.0:
            forecast = forecast.scale(scale)

        print(
            f"Forecast expected count: {forecast.event_count:.2f}"
            f" with scaling parameter: {time_horizon:.1f}")

        self.forecasts[tstring] = forecast

    def forecast_from_func(self, start_date: datetime, end_date: datetime,
                           **kwargs) -> None:
        raise NotImplementedError('TBI for time-dependent models')

    def to_dict(self, excluded=('forecasts', 'reg')):
        """

        Returns:
            Dictionary with relevant attributes. Model can be reinstantiated
            from this dict

        """
        out = {'path': self.path}

        for k, v in self.__dict__.items():
            if k not in excluded and v:
                out[k] = v
        return out

    @classmethod
    def from_dict(cls, record: dict, **kwargs):
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
        return cls(name=name, **record[name], **kwargs)
